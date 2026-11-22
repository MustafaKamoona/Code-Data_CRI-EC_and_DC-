"""Read-only diagnostics of the DC-CRI coupling weight (omega_C* = 0 in all LOSO folds).

Uses only the saved calibration data results/dc_cri_calibration_trajectories.npz
(written by notebooks/13_dc_cri_loso.ipynb), results/results_dc_cri_loso_folds.csv,
and the unmodified cri_microgrid.dc_calibration functions. No simulation is run.

Sections of the printed output:
  1. Reproduction of the stored fold weights; exact simplex-QP optimum by KKT support
     enumeration; 20-start SLSQP; CI reduced cost; smallest eigenvalue of the QP matrix.
  2. Per-scenario class composition (z^(H), T_H = 0.5 s) and indicator means.
  3. Pooled single-indicator discrimination (AUC, Cohen's d).
  4. Redundancy: correlations; OLS R^2 of CI on (RI, SMI, AI, NCI); AUC of the residual.
  5. Class-balanced logistic regression on the pooled data (in-sample AUC, with and
     without CI) and the LOSO held-out AUC per scenario.
  6. Objective cost of forcing omega_C > 0.
  7. Sensitivity of omega_C* to CI rescaling and to lambda_H.
  8. CI statistics per scenario.

Usage (from the repository root):  python scripts/ci_weight_diagnostic.py [repo_root]
The recorded output is results/ci_weight_diagnostic_output.txt.
"""
import itertools, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cri_microgrid import dc_calibration as dcc

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
F = ["RI", "CI", "SMI", "AI", "NCI"]
SC = [f"S{i}" for i in range(1, 9)]
H, LAM = 25, 2.0
d = np.load(f"{ROOT}/results/dc_cri_calibration_trajectories.npz")
G, Z, S = {}, {}, {}
for s in SC:
    gl, zl = [], []
    for seed in range(5):
        tr = dcc.TrajectoryData(s, seed, d[f"{s}_seed{seed}_Gamma"], d[f"{s}_seed{seed}_sigma"],
                                d[f"{s}_seed{seed}_viol"], 0.02)
        gl.append(tr.Gamma); zl.append(dcc.label_for_horizon(tr, H))
    G[s], Z[s] = np.vstack(gl), np.concatenate(zl)

def auc(x, z):  # P(x_safe > x_haz): >0.5 means "higher feature -> safer"
    z = z.astype(bool); r = rankdata(x); n1, n0 = (~z).sum(), z.sum()
    if n1 == 0 or n0 == 0: return np.nan
    return (r[~z].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def pool(excl=None):
    ks = [s for s in SC if s != excl]
    return np.vstack([G[s] for s in ks]), np.concatenate([Z[s] for s in ks])

def PQ(Gm, z):
    z = z.astype(bool); As, Ah = Gm[~z], Gm[z]
    P = As.T @ As / len(As) + LAM * Ah.T @ Ah / len(Ah)
    q = -2 * As.sum(0) / len(As)
    return P, q

def exact_qp(P, q):
    """Global optimum of convex QP on simplex by support enumeration (KKT)."""
    best = None
    for k in range(1, 6):
        for sup in itertools.combinations(range(5), k):
            sup = list(sup); Ps = P[np.ix_(sup, sup)]
            K = np.block([[2 * Ps, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
            try: sol = np.linalg.solve(K, np.r_[-q[sup], 1.0])
            except np.linalg.LinAlgError: continue
            w = np.zeros(5); w[sup] = sol[:k]
            if np.any(w < -1e-12): continue
            g = 2 * P @ w + q; mu = -sol[k]
            if np.all(g - g[sup].mean() >= -1e-9):
                J = w @ P @ w + q @ w + 1
                if best is None or J < best[1]: best = (w, J, g)
    return best

print("=== 1. Reproduce LOSO fold weights from saved data ===")
folds = pd.read_csv(f"{ROOT}/results/results_dc_cri_loso_folds.csv").set_index("held_out")
rows = []
for s in SC:
    Gm, z = pool(s)
    r = dcc.calibrate_weights(Gm, z, LAM)
    P, q = PQ(Gm, z); we, Je, g = exact_qp(P, q)
    sup = we > 1e-9; mu = g[sup].mean()
    # multi-start SLSQP
    rng = np.random.default_rng(0); ms = []
    for _ in range(20):
        w0 = rng.dirichlet(np.ones(5))
        rr = dcc.calibrate_weights(Gm, z, LAM, omega0=w0); ms.append(rr.omega)
    ms = np.array(ms)
    rows.append(dict(held_out=s, max_abs_diff_vs_csv=np.abs(r.omega - folds.loc[s, ["omega_" + c for c in "RCSAN"]].values).max(),
                     max_abs_diff_vs_exactQP=np.abs(r.omega - we).max(), omega_C_exact=we[1],
                     multistart_max_omegaC=ms[:, 1].max(), multistart_spread=np.abs(ms - we).max(),
                     reduced_cost_CI=g[1] - mu, eig_min_P=np.linalg.eigvalsh(P).min(),
                     success=r.success))
print(pd.DataFrame(rows).to_string(float_format=lambda x: f"{x:.3g}"))

print("\n=== 2. Per-scenario class composition and CI statistics (T_H=0.5 s labels, 5 seeds) ===")
rows = []
for s in SC:
    g, z = G[s], Z[s].astype(bool)
    row = dict(scen=s, n=len(z), haz_frac=z.mean())
    for i, f in enumerate(F):
        row[f"{f}_safe"] = g[~z, i].mean() if (~z).any() else np.nan
        row[f"{f}_haz"] = g[z, i].mean() if z.any() else np.nan
    row["CI_sd_safe"] = g[~z, 1].std() if (~z).any() else np.nan
    row["AUC_CI_within"] = auc(g[:, 1], z.astype(int))
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string())

Gall, zall = pool(None)
print("\n=== 3. Pooled (all 8 scenarios) per-feature discrimination ===")
rows = []
for i, f in enumerate(F):
    x = Gall[:, i]; zb = zall.astype(bool)
    J1 = np.mean((1 - x[~zb]) ** 2) + LAM * np.mean(x[zb] ** 2)
    rows.append(dict(feature=f, mean_safe=x[~zb].mean(), mean_haz=x[zb].mean(),
                     cohen_d=(x[~zb].mean() - x[zb].mean()) / np.sqrt(0.5 * (x[~zb].var() + x[zb].var())),
                     AUC_pooled=auc(x, zall),
                     AUC_min_over_folds=min(auc(pool(s)[0][:, i], pool(s)[1]) for s in SC),
                     AUC_max_over_folds=max(auc(pool(s)[0][:, i], pool(s)[1]) for s in SC),
                     single_feature_objective=J1))
print(pd.DataFrame(rows).round(3).to_string())

print("\n=== 4. Redundancy: correlations (pooled) ===")
print("Pearson:\n", pd.DataFrame(np.corrcoef(Gall.T), index=F, columns=F).round(3).to_string())
print("Spearman:\n", pd.DataFrame(spearmanr(Gall).correlation, index=F, columns=F).round(3).to_string())
X = np.c_[np.ones(len(Gall)), Gall[:, [0, 2, 3, 4]]]
beta, *_ = np.linalg.lstsq(X, Gall[:, 1], rcond=None)
res = Gall[:, 1] - X @ beta
print(f"R^2 of CI regressed on (RI,SMI,AI,NCI): {1 - res.var() / Gall[:, 1].var():.3f}")
print(f"AUC of CI residual (part of CI not explained by others): {auc(res, zall):.3f}")

print("\n=== 5. Incremental predictive value (logistic regression, pooled, class-balanced) ===")
def logit_fit(X, z):
    z = z.astype(float); w_s = np.where(z == 1, 0.5 / z.mean(), 0.5 / (1 - z.mean()))
    Xb = np.c_[np.ones(len(X)), X]
    def nll(b):
        t = Xb @ b; return np.sum(w_s * (np.logaddexp(0, t) - z * t)) / len(z)
    def grad(b):
        p = 1 / (1 + np.exp(-(Xb @ b))); return Xb.T @ (w_s * (p - z)) / len(z)
    r = minimize(nll, np.zeros(Xb.shape[1]), jac=grad, method="L-BFGS-B")
    return r.x, r.fun, auc(-(Xb @ r.x), z)
for name, cols in [("RI,SMI,AI,NCI", [0, 2, 3, 4]), ("all five", [0, 1, 2, 3, 4]), ("CI only", [1])]:
    b, f, a = logit_fit(Gall[:, cols], zall)
    print(f"  {name:<14s} bal-NCE={f:.4f}  AUC={a:.4f}  coefs={np.round(b, 3)}")
# LOSO version: train on 7 scenarios, test AUC on the held-out one (where both classes exist)
print("  LOSO held-out AUC (4 feat / 5 feat):")
for s in SC:
    Gm, z = pool(s); gt, zt = G[s], Z[s]
    if zt.min() == zt.max(): print(f"    {s}: single-class, n/a"); continue
    b4, _, _ = logit_fit(Gm[:, [0, 2, 3, 4]], z); b5, _, _ = logit_fit(Gm, z)
    a4 = auc(-(np.c_[np.ones(len(gt)), gt[:, [0, 2, 3, 4]]] @ b4), zt)
    a5 = auc(-(np.c_[np.ones(len(gt)), gt] @ b5), zt)
    print(f"    {s}: {a4:.4f} / {a5:.4f}")

print("\n=== 6. Forced omega_C (pooled, all 8): objective cost and score discrimination ===")
P, q = PQ(Gall, zall); w_opt, J_opt, _ = exact_qp(P, q)
print("unconstrained-on-CI optimum (all 8 pooled):", np.round(w_opt, 4), f"J={J_opt:.5f}",
      f"AUC(score)={auc(Gall @ w_opt, zall):.4f}")
for wc in [0.05, 0.10, 0.20, 0.30]:
    # minimize over the remaining 4 with omega_C fixed: substitute
    def J(v):
        w = np.r_[v[0], wc, v[1:]]; return w @ P @ w + q @ w + 1
    r = minimize(J, np.full(4, (1 - wc) / 4), method="SLSQP", bounds=[(0, 1)] * 4,
                 constraints=[{"type": "eq", "fun": lambda v: v.sum() - (1 - wc)}], options={"ftol": 1e-14})
    w = np.r_[r.x[0], wc, r.x[1:]]
    print(f"  omega_C={wc:.2f}: J={r.fun:.5f} (+{100*(r.fun-J_opt)/J_opt:.2f}%), AUC(score)={auc(Gall @ w, zall):.4f}, w={np.round(w,3)}")
w_ec = np.array([0.20, 0.10, 0.30, 0.15, 0.25])
print(f"  EC-CRI weights: J={w_ec @ P @ w_ec + q @ w_ec + 1:.5f}, AUC(score)={auc(Gall @ w_ec, zall):.4f}")
w_ec0 = w_ec.copy(); w_ec0[1] = 0; w_ec0 /= w_ec0.sum()
print(f"  EC-CRI with omega_C->0 renorm: AUC(score)={auc(Gall @ w_ec0, zall):.4f}")

print("\n=== 7. Hypothesis test: is the zero partly a LEVEL (offset) effect of the objective? ===")
# The objective targets omega^T Gamma = 1 on safe samples; a feature whose safe-class
# level is far below 1 is penalized even if it discriminates. Rescale CI by its pooled
# safe-class 95th percentile (clipped to [0,1]) -- a diagnostic redefinition only.
for lab, scale in [("CI / p95_safe", np.percentile(Gall[~zall.astype(bool), 1], 95)),
                   ("CI / max", Gall[:, 1].max())]:
    out = []
    for s in SC:
        Gm, z = pool(s); Gm = Gm.copy(); Gm[:, 1] = np.clip(Gm[:, 1] / scale, 0, 1)
        P2, q2 = PQ(Gm, z); out.append(exact_qp(P2, q2)[0][1])
    print(f"  {lab} (scale={scale:.3f}): omega_C* per fold = {np.round(out, 3)}")
# min-max on safe range
out = []
for s in SC:
    Gm, z = pool(s); Gm = Gm.copy()
    lo, hi = np.percentile(Gm[:, 1], [5, 95]); Gm[:, 1] = np.clip((Gm[:, 1] - lo) / (hi - lo), 0, 1)
    P2, q2 = PQ(Gm, z); out.append(exact_qp(P2, q2)[0][1])
print(f"  CI min-max (p5..p95): omega_C* per fold = {np.round(out, 3)}")
# lambda_H sensitivity
for lam in [1.0, 2.0, 5.0, 10.0]:
    LAM_s = LAM; LAM = lam
    out = [exact_qp(*PQ(*pool(s)))[0][1] for s in SC]
    LAM = LAM_s
    print(f"  lambda_H={lam}: omega_C* per fold = {np.round(out, 4)}")

print("\n=== 8. CI behaviour in S4 vs S1 and at CI==0 (sensitivity fallback) ===")
for s in SC:
    g = G[s]; print(f"  {s}: CI mean={g[:,1].mean():.3f} p5={np.percentile(g[:,1],5):.3f} p95={np.percentile(g[:,1],95):.3f}"
                    f"  frac(CI<0.05)={np.mean(g[:,1]<0.05):.3f}  corr(CI,RI)={np.corrcoef(g[:,0],g[:,1])[0,1] if g[:,0].std()>0 else np.nan:.3f}")
