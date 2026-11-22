"""Evaluation helpers for the RTA/CBF benchmark (notebook 15).
Reuses -- never redefines -- the existing scenario builders,
hazard labels and metric functions.

Two evaluation protocols, never mixed:
  Protocol A ("EC protocol", as in notebooks 08-10): instantaneous
    mechanism-based ground truth z_k^unsafe (validation.compute_unsafe_truth).
  Protocol B ("LOSO protocol", as in notebook 13): finite-horizon
    hard-limit label z_k^(H), H = 25 (T_H = 0.5 s); DC-CRI uses the per-fold
    parameters already saved in results/results_dc_cri_loso_folds.csv,
    calibrated without the held-out scenario. RTA and CBF use one fixed
    configuration for every fold (selected only on S10 development trials).
"""
from typing import Dict, Optional
import numpy as np
import pandas as pd

from .config import (PlantParams, ControllerParams, ConstraintParams, CRIParams, ProbeParams,
                     ProfileParams, SimulationParams)
from . import scenarios, sim, metrics, validation
from . import dc_calibration as dcc
from . import benchmark_sim as bsim
from .baselines_rta_cbf import SimplexRTASupervisor, DiscreteCBFFilter, RTAParams, CBFParams

SCENARIO_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
NOISY = {"S6", "S8"}
H_PRINCIPAL = 25


def scenario_builders(pp: PlantParams, prof: ProfileParams):
    """Identical to notebooks 08/09/13."""
    return {
        "S1": lambda: scenarios.build_S1(pp, prof, T_end=40.0),
        "S2": lambda: scenarios.build_S2(pp, prof, T_end=40.0),
        "S3": lambda: scenarios.build_S3(pp, prof, T_end=40.0),
        "S4": lambda: scenarios.build_S4(pp, prof, T_end=40.0),
        "S5": lambda: scenarios.build_S5(pp, prof, ramp_kind="fast", T_end=40.0),
        "S6": lambda: scenarios.build_S6(pp, prof, noise_level_pct=1.0, T_end=40.0),
        "S7": lambda: scenarios.build_S7(pp, prof, T_end=40.0, kind="vsc2_outage"),
        "S8": lambda: scenarios.build_S8(pp, prof, T_end=50.0),
    }


def load_dc_fold_params(path: str, crip_template: CRIParams) -> Dict[str, CRIParams]:
    df = pd.read_csv(path).set_index("held_out")
    out = {}
    for s, r in df.iterrows():
        out[s] = CRIParams(
            N=crip_template.N, Na=crip_template.Na, lambda_s=crip_template.lambda_s,
            r0=crip_template.r0, a0=crip_template.a0, sigma_y0_sq=crip_template.sigma_y0_sq,
            epsilon_0s=crip_template.epsilon_0s,
            omega_R=float(r.omega_R), omega_C=float(r.omega_C), omega_S=float(r.omega_S),
            omega_A=float(r.omega_A), omega_N=float(r.omega_N),
            eta_A=float(r.eta_A), eta_M=float(r.eta_M), eta_D=float(r.eta_D),
            eta_S=crip_template.eta_S, eta_N=crip_template.eta_N,
            beta_min=crip_template.beta_min, beta_max=crip_template.beta_max,
            kappa_beta=crip_template.kappa_beta, kappa_min=crip_template.kappa_min,
            delta_max=crip_template.delta_max, epsilon_regularization=crip_template.epsilon_regularization)
    return out


class FallbackOnlySupervisor:
    """Reference only (not a supervisor under test): s_k = 0 always, i.e. the
    fallback droop law drives the plant for the whole run."""
    name = "Fallback-only"
    def reset(self): pass
    def step(self, k, t, y_meas, u_applied, u_safe, u_candidate):
        return dict(s=0, kappa=1.0, cls="Reject", u_override=None)


def exceedance(lg: Dict, cons: ConstraintParams) -> np.ndarray:
    """Per-sample worst hard-limit exceedance (pu, >= 0) on the TRUE signals:
    max over V1, V2, Vdc, I1, I2 of the distance outside the admissible band."""
    V1, V2, Vdc = lg["V1_true"], lg["V2_true"], lg["Vdc_true"]
    I1, I2 = lg["I1_true"], lg["I2_true"]
    e = np.stack([cons.Vac_min - V1, V1 - cons.Vac_max, cons.Vac_min - V2, V2 - cons.Vac_max,
                  cons.Vdc_min - Vdc, Vdc - cons.Vdc_max, I1 - cons.I_max, I2 - cons.I_max])
    return np.maximum(0.0, e.max(axis=0))


class Bench:
    def __init__(self, rta_prm: Optional[RTAParams] = None, cbf_prm: Optional[CBFParams] = None):
        self.pp, self.cp, self.cons = PlantParams(), ControllerParams(), ConstraintParams()
        self.crip, self.probep, self.prof = CRIParams(), ProbeParams(), ProfileParams()
        self.builders = scenario_builders(self.pp, self.prof)
        self.rta_prm, self.cbf_prm = rta_prm or RTAParams(), cbf_prm or CBFParams()

    def run(self, method: str, scen: str, seed: int, crip_dc: Optional[CRIParams] = None) -> Dict:
        sc = self.builders[scen]()
        simp = SimulationParams(T_end=sc.T_end, seed=seed)
        if method == "Direct":
            simp.strategy = "direct"
            lg = sim.run_closed_loop(sc, self.pp, self.cp, self.cons, self.crip, self.probep, simp)
        elif method == "Hard-limit":
            simp.strategy = "threshold"
            lg = sim.run_closed_loop(sc, self.pp, self.cp, self.cons, self.crip, self.probep, simp)
        elif method == "EC-CRI":
            simp.strategy = "cri"
            lg = sim.run_closed_loop(sc, self.pp, self.cp, self.cons, self.crip, self.probep, simp)
        elif method == "DC-CRI":
            assert crip_dc is not None
            simp.strategy = "cri"
            lg = sim.run_closed_loop(sc, self.pp, self.cp, self.cons, crip_dc, self.probep, simp)
        elif method == "RTA":
            lg = bsim.run_closed_loop_supervised(sc, self.pp, self.cp, self.cons, simp,
                                                 SimplexRTASupervisor(self.pp, self.cp, self.cons, self.rta_prm))
        elif method == "CBF":
            lg = bsim.run_closed_loop_supervised(sc, self.pp, self.cp, self.cons, simp,
                                                 DiscreteCBFFilter(self.pp, self.cons, self.cbf_prm))
        elif method == "Fallback-only":
            lg = bsim.run_closed_loop_supervised(sc, self.pp, self.cp, self.cons, simp, FallbackOnlySupervisor())
        else:
            raise ValueError(method)
        lg["_scenario_obj"] = sc
        return lg

    # ------------------------------------------------------------------
    def authority_signal(self, lg: Dict, method: str) -> np.ndarray:
        """s used for FAR/N_FA/A_c. For every method except CBF this is the
        logged s_k. For CBF it is s^unf_k = 1 iff the candidate command was
        applied UNMODIFIED (|u_applied - u_cand|_inf <= tol): a filtered
        command is not counted as unrestricted candidate authority."""
        if method == "CBF":
            return 1.0 - lg["diag_filtered"]
        return np.asarray(lg["s"], dtype=float)

    def common(self, lg: Dict, method: str, s: np.ndarray) -> Dict:
        cons, Ts = self.cons, self.pp.Ts
        viol = dcc.hard_limit_violation(lg, cons)
        exc = exceedance(lg, cons)
        out = dict(
            A_c=metrics.controller_availability(s),
            E_V=metrics.voltage_deviation(lg["V1"], lg["V2"], self.pp.Vac_nom),
            E_I=metrics.current_violation(lg["I1"], lg["I2"], cons.I_max),
            T_on_s=metrics.active_duration(s, Ts),
            n_violation_true=int(np.sum(viol)),
            violation_true_pct=100.0 * float(np.mean(viol)),
            max_exceedance_pu=float(exc.max()),
            int_exceedance_pu_s=float(exc.sum() * Ts),
            n_switch=int(np.sum(np.abs(np.diff(s)) > 0.5)),
        )
        if method == "RTA":
            out["rta_n_interventions"] = int(np.sum(np.diff(lg["s"]) < -0.5))   # 1 -> 0 switches
            out["rta_monitor_alarm_pct"] = 100.0 * float(np.nanmean(1 - lg["diag_monitor_pass"][1:]))
            out["sup_time_mean_ms"] = 1e3 * float(np.nanmean(lg["diag_sup_time_s"]))
            out["sup_time_max_ms"] = 1e3 * float(np.nanmax(lg["diag_sup_time_s"]))
        if method == "CBF":
            st = lg["diag_cbf_status"]
            out["cbf_filtered_pct"] = 100.0 * float(np.nanmean(lg["diag_filtered"]))
            out["cbf_mean_dev_inf"] = float(np.nanmean(lg["diag_filter_dev_inf"]))
            out["cbf_relaxed_pct"] = 100.0 * float(np.mean(st == 1))
            out["cbf_solver_fail_pct"] = 100.0 * float(np.mean(st >= 10))
            out["cbf_worst_resid"] = float(np.nanmin(lg["diag_cbf_resid_after"] + lg["diag_cbf_slack"])) \
                if np.any(np.isfinite(lg.get("diag_cbf_resid_after", np.array([np.nan])))) else np.nan
            out["A_c_nominal"] = float(np.mean(lg["s"]))      # always 1 by construction
            out["sup_time_mean_ms"] = 1e3 * float(np.nanmean(lg["diag_sup_time_s"]))
            out["sup_time_max_ms"] = 1e3 * float(np.nanmax(lg["diag_sup_time_s"]))
        return out

    def metrics_A(self, lg: Dict, method: str) -> Dict:
        sc = lg["_scenario_obj"]
        z = validation.compute_unsafe_truth(lg, self.cons, noise_level_pct=sc.noise_cfg.get("level_pct", 0.0),
                                            noise_window=sc.noise_window)
        s = self.authority_signal(lg, method)
        out = dict(n_hazard=int(z.sum()),
                   N_FA=metrics.false_activation_count(s.astype(int), z),
                   active_in_hazard_pct=100.0 * metrics.unsafe_active_fraction(s, z),
                   safe_availability_pct=metrics.safe_availability_pct(s, z))
        out.update(self.common(lg, method, s))
        return out

    def metrics_B(self, lg: Dict, method: str, H: int = H_PRINCIPAL) -> Dict:
        zH = dcc.finite_horizon_label(lg, self.cons, H)
        s = self.authority_signal(lg, method)
        out = dict(n_hazard_H=int(zH.sum()),
                   N_FA_H=metrics.false_activation_count_horizon(s.astype(int), zH),
                   FAR=metrics.far(s, zH), FDR=metrics.fdr(s, zH))
        out.update(self.common(lg, method, s))
        return out
