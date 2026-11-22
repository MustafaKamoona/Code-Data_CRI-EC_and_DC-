"""One-at-a-time EC-CRI weight-sensitivity study.

Usage (from the repository root):  python scripts/ec_cri_weight_sensitivity.py

Extends Sweep 3 of notebooks/09_parameter_sensitivity.ipynb (cells 12-13)
with the same setup: identical scenario builders, seed, controller /
plant / constraint / probing parameters, strategy ("cri"), and EC-CRI
ground-truth hazard label (validation.compute_unsafe_truth with the same
arguments). Only the five CRI weights change; thresholds and normalization
constants are the CRIParams() defaults.

Configurations (12): Nominal EC-CRI, Equal weights (1/5), and each omega_i
scaled by 0.8 and 1.2 with the full vector renormalized to unit sum.

Outputs (results/ directory; the script refuses to overwrite existing files):
  r1c5_weight_sensitivity_configs.csv      -- normalized weight vectors
  r1c5_weight_sensitivity_perscenario.csv  -- per (config, scenario) metrics
  r1c5_weight_sensitivity_aggregate.csv    -- mean over S1-S8 per config

Metrics use the EC-CRI instantaneous ground truth z_k^unsafe ONLY (not the
finite-horizon LOSO label z_k^(H)).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cri_microgrid.config import (PlantParams, ControllerParams, ConstraintParams, CRIParams,  # noqa: E402
                                  ProbeParams, ProfileParams, SimulationParams)
from cri_microgrid import scenarios, sim, metrics, validation  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
OUT_CFG = os.path.join(RESULTS_DIR, "r1c5_weight_sensitivity_configs.csv")
OUT_PER = os.path.join(RESULTS_DIR, "r1c5_weight_sensitivity_perscenario.csv")
OUT_AGG = os.path.join(RESULTS_DIR, "r1c5_weight_sensitivity_aggregate.csv")
for p in (OUT_CFG, OUT_PER, OUT_AGG):
    if os.path.exists(p):
        raise SystemExit(f"Refusing to overwrite existing file: {p}")

pp = PlantParams()
cp = ControllerParams()
cons = ConstraintParams()
probep = ProbeParams()
prof = ProfileParams()
SEED = 1  # same as notebook 09

nominal = CRIParams()
NAMES = ["omega_R", "omega_C", "omega_S", "omega_A", "omega_N"]
w_nom = nominal.weights()

print("Thresholds used by code (CRIParams defaults):",
      dict(eta_A=nominal.eta_A, eta_M=nominal.eta_M, eta_D=nominal.eta_D,
           eta_S=nominal.eta_S, eta_N=nominal.eta_N))
print("Nominal weights:", w_nom)

configs = {"Nominal": w_nom.copy(), "Equal": np.full(5, 0.2)}
for i, n in enumerate(NAMES):
    for mult in (0.8, 1.2):
        w = w_nom.copy()
        w[i] *= mult
        configs[f"{n} x{mult}"] = w / w.sum()

# --- identical to notebook 09, cell 13 ---------------------------------------
builders = {
    "S1": lambda: scenarios.build_S1(pp, prof, T_end=40.0),
    "S2": lambda: scenarios.build_S2(pp, prof, T_end=40.0),
    "S3": lambda: scenarios.build_S3(pp, prof, T_end=40.0),
    "S4": lambda: scenarios.build_S4(pp, prof, T_end=40.0),
    "S5": lambda: scenarios.build_S5(pp, prof, ramp_kind="fast", T_end=40.0),
    "S6": lambda: scenarios.build_S6(pp, prof, noise_level_pct=1.0, T_end=40.0),
    "S7": lambda: scenarios.build_S7(pp, prof, T_end=40.0, kind="vsc2_outage"),
    "S8": lambda: scenarios.build_S8(pp, prof, T_end=50.0),
}

cfg_rows, per_rows = [], []
t0 = time.time()
for cname, w in configs.items():
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12
    crip = CRIParams(omega_R=w[0], omega_C=w[1], omega_S=w[2], omega_A=w[3], omega_N=w[4])
    cfg_rows.append(dict(config=cname, **{n: float(v) for n, v in zip(NAMES, w)}))
    for sname, b in builders.items():
        sc = b()
        simp = SimulationParams(T_end=sc.T_end, seed=SEED, strategy="cri")
        lg = sim.run_closed_loop(sc, pp, cp, cons, crip, probep, simp)
        z = validation.compute_unsafe_truth(lg, cons, noise_level_pct=sc.noise_cfg.get("level_pct", 0.0),
                                            noise_window=sc.noise_window)
        s = lg["s"]
        cls = lg["cls_code"]
        cri = lg["CRI"]
        n_haz = int(np.sum(z == 1))
        sa = metrics.safe_availability_pct(s, z)
        per_rows.append(dict(
            config=cname, scenario=sname, n_samples=len(s), n_hazard=n_haz,
            N_FA=metrics.false_activation_count(s, z),
            active_in_hazard_pct=metrics.unsafe_active_fraction(s, z) * 100.0 if n_haz > 0 else np.nan,
            A_c=metrics.controller_availability(s),
            safe_availability_pct=sa,
            false_deactivation_pct=(100.0 - sa) if np.isfinite(sa) else np.nan,
            mean_CRI=float(np.mean(cri)),
            mean_CRI_after_window=float(np.mean(cri[crip.N:])),
            min_CRI_after_window=float(np.min(cri[crip.N:])),
            pct_Activate=100.0 * np.mean(cls == 3), pct_Monitor=100.0 * np.mean(cls == 2),
            pct_Delay=100.0 * np.mean(cls == 1), pct_Reject=100.0 * np.mean(cls == 0),
            mean_kappa=float(np.mean(lg["kappa"])),
        ))
    print(f"  {cname:<14s} done ({time.time() - t0:.1f} s)")

cfg_df = pd.DataFrame(cfg_rows)
per_df = pd.DataFrame(per_rows)

# Aggregate: mean over the 8 scenarios, same convention as notebook 09
# (active-in-hazard averaged over all 8 scenarios with non-hazard ones = 0,
# exactly as unsafe_active_fraction returns 0 there in notebook 09).
per_df_nb = per_df.copy()
per_df_nb["aih_nb09"] = per_df_nb["active_in_hazard_pct"].fillna(0.0)
agg = per_df_nb.groupby("config", sort=False).agg(
    mean_N_FA=("N_FA", "mean"),
    mean_active_in_hazard_pct=("aih_nb09", "mean"),
    mean_A_c=("A_c", "mean"),
    mean_safe_availability_pct=("safe_availability_pct", "mean"),
    mean_false_deactivation_pct=("false_deactivation_pct", "mean"),
).reset_index()
haz = per_df[per_df["n_hazard"] > 0].groupby("config", sort=False).agg(
    hazardous_scen_mean_N_FA=("N_FA", "mean"),
    hazardous_scen_mean_A_c=("A_c", "mean")).reset_index()
safe = per_df[per_df["scenario"].isin(["S1", "S2", "S4", "S6"])].groupby("config", sort=False).agg(
    safe_scen_mean_A_c=("A_c", "mean")).reset_index()
s1 = per_df[per_df["scenario"] == "S1"][["config", "mean_CRI", "mean_CRI_after_window", "min_CRI_after_window"]]
s1 = s1.rename(columns={"mean_CRI": "S1_mean_CRI", "mean_CRI_after_window": "S1_mean_CRI_after_window",
                        "min_CRI_after_window": "S1_min_CRI_after_window"})
agg = agg.merge(haz, on="config").merge(safe, on="config").merge(s1, on="config").merge(cfg_df, on="config")
agg["S1_headroom_eta_A"] = agg["S1_mean_CRI_after_window"] - nominal.eta_A
agg["S1_headroom_eta_M"] = agg["S1_mean_CRI_after_window"] - nominal.eta_M
nom = agg.loc[agg["config"] == "Nominal"].iloc[0]
agg["N_FA_pct_change"] = 100.0 * (agg["mean_N_FA"] - nom["mean_N_FA"]) / nom["mean_N_FA"]
agg["AIH_pp_change"] = agg["mean_active_in_hazard_pct"] - nom["mean_active_in_hazard_pct"]
agg["A_c_pp_change"] = 100.0 * (agg["mean_A_c"] - nom["mean_A_c"])

cfg_df.to_csv(OUT_CFG, index=False)
per_df.to_csv(OUT_PER, index=False)
agg.to_csv(OUT_AGG, index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(agg.round(4).to_string())
print(f"\nTotal runtime {time.time() - t0:.1f} s")
print("Saved:", OUT_CFG, OUT_PER, OUT_AGG, sep="\n  ")
