"""CRI closed loop with an optional Measurement Validity Gate (MVG) and optional
measurement-fault injection (notebook 16). sim.py,
cri.py, sensitivity.py and controllers.py are used unmodified.

`run_closed_loop_mvg` reproduces the 'cri' path of sim.run_closed_loop
statement-for-statement. With gate=None and fault=None, or with the gate
enabled and no fault injected, it is intended to be numerically identical to
sim.run_closed_loop(strategy='cri') -- verified in notebook 16, not assumed.

Measurement pipeline per sample k:
  plant measurement y_k (true + simulated noise)
    -> fault injector (optional)  -> delivered (y_k^del, tau_k)
    -> MVG (optional)              -> valid / invalid / recovering
    -> CRI and candidate receive y_k^del only when the whole sample is valid;
       the fallback receives each channel's current value when that channel is
       usable for it (see mvg.py) and otherwise the channel's last usable value.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .config import (PlantParams, ControllerParams, ConstraintParams, CRIParams, ProbeParams,
                     SimulationParams, DEFAULT_D_u, DEFAULT_D_y)
from .plant import HybridACDCPlant
from .controllers import FallbackDroopController, OFOVSCController, blend_command
from .cri import CRISupervisor
from . import probing as pb
from .scenarios import Scenario
from .sim import LOG_FIELDS, _CLASS_CODE, _active_noise_cfg, _u_bounds
from .mvg import MeasurementValidityGate, VALID, INVALID, RECOVERING

_STATE_CODE = {VALID: 2, RECOVERING: 1, INVALID: 0}


# =============================================================================
# Measurement-fault injection (test harness only)
# =============================================================================
@dataclass
class Fault:
    kind: str                 # 'nan' | 'dropout' | 'stale' | 'delay' | 'frozen' | 'stuck_value' | 'offset'
    t0: float
    t1: float                 # active for t0 <= t < t1
    channel: Optional[int] = None
    value: float = 0.0        # stuck_value / offset magnitude
    delay_steps: int = 0      # 'delay'


class FaultInjector:
    """Transforms the plant measurement into the delivered (y, timestamp).
    Fresh, fault-free delivery: (y_k, t_k). Faults:
      nan         : channel -> NaN
      dropout     : whole vector -> NaN (lost packet)
      stale       : the last delivered packet is repeated (values AND timestamp)
      delay       : packet from delay_steps samples earlier, with its own (old) timestamp
      frozen      : channel holds the value it had at fault onset (fresh timestamp)
      stuck_value : channel reads a constant `value` (fresh timestamp)
      offset      : channel + value (fresh timestamp; outlier when t1 - t0 = one sample)
    """
    def __init__(self, faults: List[Fault], timestamps: bool = True):
        self.faults, self.timestamps = faults, timestamps
        self.hist = deque(maxlen=64)
        self.last_delivered = None
        self.frozen_val = {}

    def apply(self, k: int, t: float, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        self.hist.append((y.copy(), t))
        y_d, ts = y.copy(), t
        for i, f in enumerate(self.faults):
            if not (f.t0 - 1e-9 <= t < f.t1 - 1e-9):
                continue
            if f.kind == "nan":
                y_d[f.channel] = np.nan
            elif f.kind == "dropout":
                y_d[:] = np.nan
            elif f.kind == "stale" and self.last_delivered is not None:
                y_d, ts = self.last_delivered[0].copy(), self.last_delivered[1]
            elif f.kind == "delay":
                j = max(0, len(self.hist) - 1 - f.delay_steps)
                y_d, ts = self.hist[j][0].copy(), self.hist[j][1]
            elif f.kind == "frozen":
                if i not in self.frozen_val:
                    self.frozen_val[i] = float(self.last_delivered[0][f.channel]) if self.last_delivered is not None else float(y[f.channel])
                y_d[f.channel] = self.frozen_val[i]
            elif f.kind == "stuck_value":
                y_d[f.channel] = f.value
            elif f.kind == "offset":
                y_d[f.channel] = y_d[f.channel] + f.value
        self.last_delivered = (y_d.copy(), ts)
        return y_d, (ts if self.timestamps else None)


# =============================================================================
# Closed loop
# =============================================================================
def run_closed_loop_mvg(scenario: Scenario, plant_params: PlantParams, ctrl_params: ControllerParams,
                        cons: ConstraintParams, cri_params: CRIParams, probe_params: ProbeParams,
                        sim_params: SimulationParams, gate: Optional[MeasurementValidityGate] = None,
                        fault: Optional[FaultInjector] = None, probing_enabled: bool = True,
                        catch_exceptions: bool = False) -> Dict[str, np.ndarray]:
    Ts = plant_params.Ts
    n = int(np.floor(scenario.T_end / Ts)) + 1
    t_axis = np.arange(n) * Ts

    rng = np.random.default_rng(sim_params.seed)
    plant = HybridACDCPlant(plant_params, noise_rng=rng)
    fallback = FallbackDroopController(ctrl_params, cons, plant_params)
    candidate = OFOVSCController(ctrl_params, cons, plant_params)
    cri_sup = CRISupervisor(cri_params, D_u=DEFAULT_D_u, D_y=DEFAULT_D_y)
    u_lo, u_hi = _u_bounds(cons)
    if gate is not None:
        gate.reset()

    log = {k: np.zeros(n) for k in LOG_FIELDS}
    extra = {k: np.full(n, np.nan) for k in ["mvg_valid", "mvg_state", "mvg_reset", "y_del_V1", "y_del_V2",
                                             "y_del_Vdc", "y_del_I1", "y_del_I2"]}
    reasons = [""] * n

    exo0 = scenario.exogenous_fn(0.0)
    y_meas, hidden0 = plant.measure(exo0, _active_noise_cfg(scenario, 0.0))
    u_applied = candidate.candidate_command().copy()

    natural_du_hist = deque(maxlen=cri_params.N)
    y_fb_last = None
    exception = None

    for k in range(n):
        t = t_axis[k]
        exo = scenario.exogenous_fn(t)
        gain_true = scenario.actuator_gain_fn(t)
        topo = scenario.topology_fn(t)

        # --- measurement delivery and validity gate ---
        if fault is not None:
            y_del, ts = fault.apply(k, t, y_meas)
        else:
            y_del, ts = y_meas, t
        if gate is not None:
            g = gate.check(t, y_del, ts, u_applied)
            valid, gstate = g["valid"], g["state"]
            if g["reset_cri"]:
                cri_sup.reset()
        else:
            g, valid, gstate = None, True, VALID
        # fallback input: channel-wise; unusable channels are replaced by their last usable value
        if valid:
            y_ctrl = y_del
            y_fb_last = np.array(y_del, dtype=float)
        else:
            ok_fb = g["chan_ok_fb"]
            if y_fb_last is None:
                y_fb_last = np.where(ok_fb, y_del, np.nan)
            else:
                y_fb_last = np.where(ok_fb, y_del, y_fb_last)
            y_ctrl = y_fb_last if np.all(np.isfinite(y_fb_last)) else None

        try:
            if y_ctrl is not None:
                u_safe = fallback.command(y_ctrl)
            else:                                       # no usable value yet for some channel
                u_safe = np.clip(u_applied, u_lo, u_hi)
            u_candidate = candidate.candidate_command()
            delta_nom = candidate.propose_nominal_step(y_del) if valid else np.zeros(4)

            probe_bar = 0.0
            probe_vec = np.zeros(4)

            if valid:
                cri_out = cri_sup.step(u_applied, y_del, cons)
                s_k, kappa_k, cls = cri_out.s, cri_out.kappa, cri_out.cls
                dt_flag = cri_out.dt_flag
                if gstate == RECOVERING:
                    s_k, dt_flag = 0, 1
            else:
                cri_out = None
                s_k, kappa_k, cls, dt_flag = 0, 0.0, "Reject", 1

            u_intended = blend_command(u_safe, u_candidate, s_k, kappa_k)
            natural_du_hist.append((u_intended - u_applied) / DEFAULT_D_u)

            if probing_enabled and cri_out is not None:
                smi_now = cri_out.SMI
                du_hist = np.stack(natural_du_hist, axis=1) if len(natural_du_hist) else np.zeros((4, 0))
                if pb.need_probe(du_hist, probe_params.min_natural_excitation):
                    probe_vec, probe_bar = pb.probing_perturbation(
                        k, u_intended, smi_now, cri_params.delta_max,
                        probe_params.period_steps, np.array(probe_params.channel_scale),
                        u_lo, u_hi)

            u_next = np.clip(u_intended + probe_vec, u_lo, u_hi)
            candidate.shadow_update(delta_nom, kappa_k)
        except Exception as e:                          # only used to document unprotected behaviour
            if not catch_exceptions:
                raise
            exception = f"k={k} t={t:.2f}s {type(e).__name__}: {e}"
            n = k
            break

        y_next, hidden = plant.step(u_next, t, exo, gain_true, topo, _active_noise_cfg(scenario, t))

        log["t"][k] = t
        log["s"][k] = s_k
        log["kappa"][k] = kappa_k
        log["cls_code"][k] = _CLASS_CODE[cls]
        log["probe_bar"][k] = probe_bar
        log["probe_norm"][k] = float(np.max(np.abs(probe_vec)))
        log["dt_flag"][k] = dt_flag
        if cri_out is not None:
            for f in ["CRI", "CRI_hat", "RI", "CI", "SMI", "AI", "NCI", "beta", "epsilon_s", "rho",
                      "condition_number", "excitation_energy", "sigma_hat_sq", "a_raw"]:
                log[f][k] = getattr(cri_out, f)
        else:
            for f in ["CRI", "CRI_hat", "RI", "CI", "SMI", "AI", "NCI", "beta", "epsilon_s", "rho",
                      "condition_number", "excitation_energy", "sigma_hat_sq", "a_raw"]:
                log[f][k] = np.nan

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

        extra["mvg_valid"][k] = float(valid)
        extra["mvg_state"][k] = _STATE_CODE[gstate]
        extra["mvg_reset"][k] = float(bool(g and g["reset_cri"]))
        for j, nm in enumerate(["V1", "V2", "Vdc", "I1", "I2"]):
            extra["y_del_" + nm][k] = y_del[j]
        reasons[k] = g["reasons"] if g else ""

        u_applied = u_next
        y_meas = y_next

    if n < len(t_axis):
        for key in log:
            log[key] = log[key][:n]
        for key in extra:
            extra[key] = extra[key][:n]
        reasons = reasons[:n]
    log.update(extra)
    log["mvg_reasons"] = reasons
    log["exception"] = exception
    log["strategy"] = "cri+mvg" if gate is not None else "cri"
    log["scenario_name"] = scenario.name
    log["shat_snapshots"] = {}
    return log
