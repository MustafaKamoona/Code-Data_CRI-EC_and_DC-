"""Closed-loop runner with a pluggable supervisor, used by the RTA/CBF
benchmark (notebook 15). sim.py is not modified.

`run_closed_loop_supervised` reproduces sim.run_closed_loop's non-CRI path
statement-for-statement (same plant, fallback, OFO candidate, RNG call
order, initial command, box clipping, shadow update, and logging), but asks
a supervisor object for (s_k, kappa_k, optional command override) at every
step. With `DirectSupervisor` / `HardLimitSupervisor` it is bit-identical
to sim.run_closed_loop(strategy='direct' / 'threshold') -- verified in
notebook 15.

Supervisor interface:
    reset()
    step(k, t, y_meas, u_applied, u_safe, u_candidate) -> dict with keys
        s (0/1), kappa (float), cls (str), u_override (None or 4-array),
        plus any extra scalar diagnostics (logged as diag_<key>).
No probing is applied (as for 'direct'/'threshold' in sim.py).
"""
from typing import Dict, Optional
import numpy as np

from .config import (PlantParams, ControllerParams, ConstraintParams, SimulationParams)
from .plant import HybridACDCPlant
from .controllers import FallbackDroopController, OFOVSCController, blend_command
from .cri import microgrid_margins, compute_SMI
from .scenarios import Scenario
from .sim import LOG_FIELDS, _CLASS_CODE, _active_noise_cfg, _u_bounds


class DirectSupervisor:
    name = "Direct"
    def reset(self): pass
    def step(self, k, t, y_meas, u_applied, u_safe, u_candidate):
        return dict(s=1, kappa=1.0, cls="Activate", u_override=None)


class HardLimitSupervisor:
    name = "Hard-limit"
    def __init__(self, cons: ConstraintParams, threshold_margin: float = 0.0):
        self.cons, self.thr = cons, threshold_margin
    def reset(self): pass
    def step(self, k, t, y_meas, u_applied, u_safe, u_candidate):
        ok = min(microgrid_margins(y_meas, u_applied, self.cons)) >= self.thr
        return dict(s=1 if ok else 0, kappa=1.0, cls="Activate" if ok else "Reject", u_override=None)


def run_closed_loop_supervised(scenario: Scenario, plant_params: PlantParams, ctrl_params: ControllerParams,
                               cons: ConstraintParams, sim_params: SimulationParams, supervisor) -> Dict:
    Ts = plant_params.Ts
    n = int(np.floor(scenario.T_end / Ts)) + 1
    t_axis = np.arange(n) * Ts

    rng = np.random.default_rng(sim_params.seed)
    plant = HybridACDCPlant(plant_params, noise_rng=rng)
    fallback = FallbackDroopController(ctrl_params, cons, plant_params)
    candidate = OFOVSCController(ctrl_params, cons, plant_params)
    supervisor.reset()
    u_lo, u_hi = _u_bounds(cons)

    log = {k: np.zeros(n) for k in LOG_FIELDS}
    diag = {}

    exo0 = scenario.exogenous_fn(0.0)
    y_meas, hidden0 = plant.measure(exo0, _active_noise_cfg(scenario, 0.0))
    u_applied = candidate.candidate_command().copy()

    for k in range(n):
        t = t_axis[k]
        exo = scenario.exogenous_fn(t)
        gain_true = scenario.actuator_gain_fn(t)
        topo = scenario.topology_fn(t)

        u_safe = fallback.command(y_meas)
        u_candidate = candidate.candidate_command()
        delta_nom = candidate.propose_nominal_step(y_meas)

        out = supervisor.step(k, t, y_meas, u_applied, u_safe, u_candidate)
        s_k, kappa_k, cls = int(out["s"]), float(out["kappa"]), out["cls"]

        if out.get("u_override") is not None:
            u_intended = np.asarray(out["u_override"], dtype=float)
        else:
            u_intended = blend_command(u_safe, u_candidate, s_k, kappa_k)
        u_next = np.clip(u_intended + np.zeros(4), u_lo, u_hi)

        candidate.shadow_update(delta_nom, kappa_k)
        y_next, hidden = plant.step(u_next, t, exo, gain_true, topo, _active_noise_cfg(scenario, t))

        log["t"][k] = t
        log["s"][k] = s_k
        log["kappa"][k] = kappa_k
        log["cls_code"][k] = _CLASS_CODE[cls]
        log["probe_bar"][k] = 0.0
        log["probe_norm"][k] = 0.0
        margins = microgrid_margins(y_meas, u_applied, cons)
        log["SMI"][k] = compute_SMI(margins)
        log["CRI"][k] = float(s_k)
        log["RI"][k] = log["CI"][k] = log["AI"][k] = log["NCI"][k] = np.nan

        log["V1"][k], log["V2"][k], log["Vdc"][k] = y_meas[0], y_meas[1], y_meas[2]
        log["I1"][k], log["I2"][k] = y_meas[3], y_meas[4]
        log["P_POI"][k], log["Q_POI"][k] = y_meas[5], y_meas[6]
        log["V1_true"][k], log["V2_true"][k], log["Vdc_true"][k] = hidden["V1"], hidden["V2"], hidden["Vdc"]
        log["I1_true"][k], log["I2_true"][k] = hidden["I1"], hidden["I2"]
        log["P1_true"][k], log["Q1_true"][k] = hidden["P1"], hidden["Q1"]
        log["P2_true"][k], log["Q2_true"][k] = hidden["P2"], hidden["Q2"]
        log["P1_cmd"][k], log["Q1_cmd"][k], log["P2_cmd"][k], log["Q2_cmd"][k] = u_next
        log["P1_candidate"][k], log["Q1_candidate"][k], log["P2_candidate"][k], log["Q2_candidate"][k] = u_candidate
        log["g1_true"][k], log["g2_true"][k] = gain_true
        log["Pload1_true"][k], log["Pload2_true"][k], log["Ppv_true"][k] = exo["Pload1"], exo["Pload2"], exo["Ppv"]
        log["R1_self_true"][k], log["R2_self_true"][k] = topo.R_self[0], topo.R_self[1]
        log["Rmutual_true"][k] = topo.R_mutual

        for key, val in out.items():
            if key in ("s", "kappa", "cls", "u_override"):
                continue
            if key not in diag:
                diag[key] = np.full(n, np.nan)
            diag[key][k] = val

        u_applied = u_next
        y_meas = y_next

    for key, arr in diag.items():
        log["diag_" + key] = arr
    log["strategy"] = getattr(supervisor, "name", "custom")
    log["scenario_name"] = scenario.name
    log["shat_snapshots"] = {}
    return log
