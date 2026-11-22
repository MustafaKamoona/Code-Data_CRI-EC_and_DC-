"""Benchmark supervisors for comparing CRI with established safety supervision: a Simplex-style
runtime-assurance (RTA) switch and a discrete-time control-barrier-function
(CBF) safety filter, both built on ONE shared nominal-model predictor.

This module imports, but never modifies, the plant/controller/CRI code;
every other experiment in the repository is unaffected by it.

Information available to both baselines (same class as the CRI layer plus
the nominal model the candidate OFO already embeds):
  * the measured output y_k = [V1, V2, Vdc, I1, I2, P_POI, Q_POI] (noisy
    where the scenario injects noise),
  * the previously applied command u_{k-1},
  * the operational limits (ConstraintParams),
  * NOMINAL plant parameters (PlantParams: R/X, tau_p/tau_q/tau_v, Cdc,
    eta_conv, Ts, S_rated) -- unity actuator gain, both VSCs online,
    nominal topology.
They never see hidden truth (actuator gain, topology, loads/PV, true noise
level). Unmeasured exogenous inputs are handled by lumped-disturbance
estimates from successive measurements (offset-free prediction), held
constant over the prediction horizon.

Nominal discrete-time model (identical structure to plant.HybridACDCPlant.step
with g=1, nominal topology, both VSCs online):
  p_{k+1}  = p_k + A (sat(u_k) - p_k),           A = diag(ap, aq, ap, aq)
  V_{k+1}  = V_k + av (w + M p_k - V_k),          M = [[R11 X11 Rm Xm],[Rm Xm R22 X22]]
  E_{k+1}  = E_k + Ts (d - (p1_{k+1} + p3_{k+1})/eta),   E = Cdc Vdc^2 / 2
  I_{i,k+1}= ||(P_i,Q_i)_{k+1}|| / V_{i,k+1}
with lumped disturbances w (AC, 2-vector) and d (DC power balance, scalar).
No safety guarantee is claimed for the TRUE plant: model mismatch (actuator
gain loss, topology change, converter outage, coupling change), disturbance
variation inside the horizon, and measurement noise all violate the model
assumptions under which the RTA/CBF properties hold.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import numpy as np
from scipy.optimize import minimize

from .config import PlantParams, ControllerParams, ConstraintParams, DEFAULT_D_u
from .controllers import FallbackDroopController


# =============================================================================
# Configuration (fixed before any S1-S8 evaluation; see notebook 15, Sec. 2)
# =============================================================================
@dataclass
class PredictorParams:
    # EMA factor on measured outputs. Selected on the S10 development trials
    # (seeds 1000-1004, disjoint from S1-S8) as the value in {1.0, 0.5, 0.3}
    # minimizing the worst-case 95th-percentile 25-step Vdc prediction error.
    alpha_y: float = 0.5
    alpha_d: float = 0.1      # EMA factor on lumped-disturbance estimates (tau ~ Ts/alpha = 0.2 s)


@dataclass
class TighteningParams:
    """Constraint tightening applied by both baselines (pu)."""
    delta_V: float = 0.005    # 10% of the 0.05 pu AC half-band
    delta_dc: float = 0.008   # 10% of the 0.08 pu DC half-band
    delta_I: float = 0.02     # 2% of I_max


@dataclass
class RTAParams:
    horizon_steps: int = 25       # N_p: 0.5 s look-ahead (= T_H of the finite-horizon label)
    # M_ret: consecutive passing checks required to return authority (2 s).
    # Selected on the S10 development trials (seeds 1000-1004) from {25, 50, 100}
    # as the value minimizing total hard-limit violations (a-priori value was 25).
    return_dwell_steps: int = 100
    tightening: TighteningParams = field(default_factory=TighteningParams)


@dataclass
class CBFParams:
    # DT-CBF decay rate, h_{k+1} >= (1-gamma) h_k, 0 < gamma <= 1. Selected on the
    # S10 development trials from {0.1, 0.2, 0.5, 1.0} as the value minimizing total
    # hard-limit violations (a-priori value was 0.2). gamma = 1 enforces h_{k+1} >= 0.
    gamma: float = 1.0
    feas_tol: float = 1.0e-8      # residual tolerance used to accept a QP solution as feasible
    tightening: TighteningParams = field(default_factory=TighteningParams)
    modified_tol: float = 1.0e-3  # pu (inf-norm): |u_applied - u_cand| above this counts as "filtered"
    maxiter: int = 100
    ftol: float = 1.0e-10


# =============================================================================
# Shared nominal-model predictor
# =============================================================================
def _sat_pair(P, Q, S_max):
    S = np.hypot(P, Q)
    if S <= S_max or S < 1e-12:
        return P, Q
    return P * S_max / S, Q * S_max / S


def _sat_u(u, S_max):
    P1, Q1 = _sat_pair(u[0], u[1], S_max)
    P2, Q2 = _sat_pair(u[2], u[3], S_max)
    return np.array([P1, Q1, P2, Q2])


class NominalMicrogridPredictor:
    def __init__(self, pp: PlantParams, prm: Optional[PredictorParams] = None):
        self.pp, self.prm = pp, (prm or PredictorParams())
        self.ap = 1.0 - np.exp(-pp.Ts / pp.tau_p)
        self.aq = 1.0 - np.exp(-pp.Ts / pp.tau_q)
        self.av = 1.0 - np.exp(-pp.Ts / pp.tau_v)
        self.A = np.array([self.ap, self.aq, self.ap, self.aq])
        R11, R22 = pp.R_self
        X11, X22 = pp.X_self
        self.M = np.array([[R11, X11, pp.R_mutual, pp.X_mutual],
                           [pp.R_mutual, pp.X_mutual, R22, X22]])
        self.reset()

    def reset(self):
        self.p = None          # nominal actuator-state estimate p_k
        self.p_prev = None
        self.yf = None         # filtered measurement
        self.yf_prev = None
        self.d = None          # lumped DC disturbance
        self.w = None          # lumped AC disturbance (2,)

    def E_of(self, vdc):
        return 0.5 * self.pp.Cdc * vdc * vdc

    def vdc_of(self, E):
        return float(np.sqrt(2.0 * max(E, 1e-9) / self.pp.Cdc))

    @property
    def ready(self) -> bool:
        return self.d is not None

    def observe(self, y_meas: np.ndarray, u_prev: Optional[np.ndarray]):
        """Assimilate y_k, given the command u_{k-1} that produced it
        (u_prev=None at k=0, where p_0 is initialized from the first command)."""
        y = np.asarray(y_meas, dtype=float)
        if self.yf is None:
            self.yf = y.copy()
            return
        self.yf_prev = self.yf.copy()
        self.yf = self.yf + self.prm.alpha_y * (y - self.yf)
        self.p_prev = self.p.copy()
        self.p = self.p + self.A * (_sat_u(u_prev, self.pp.S_rated_per_vsc) - self.p)
        # one-step lumped-disturbance estimates (inverting the nominal model)
        d_inst = (self.E_of(self.yf[2]) - self.E_of(self.yf_prev[2])) / self.pp.Ts \
                 + (self.p[0] + self.p[2]) / self.pp.eta_conv
        w_inst = self.yf_prev[:2] + (self.yf[:2] - self.yf_prev[:2]) / self.av - self.M @ self.p_prev
        if self.d is None:
            self.d, self.w = d_inst, w_inst
        else:
            self.d += self.prm.alpha_d * (d_inst - self.d)
            self.w = self.w + self.prm.alpha_d * (w_inst - self.w)

    def init_actuator_state(self, u0: np.ndarray):
        if self.p is None:
            self.p = _sat_u(np.asarray(u0, dtype=float), self.pp.S_rated_per_vsc)

    def current_state(self):
        """(p_k, V_k, E_k, I_k) used as the prediction initial condition."""
        return self.p.copy(), self.yf[:2].copy(), self.E_of(self.yf[2]), self.yf[3:5].copy()

    def step_model(self, p, V, E, u):
        p_next = p + self.A * (_sat_u(u, self.pp.S_rated_per_vsc) - p)
        V_next = np.clip(V + self.av * (self.w + self.M @ p - V), 0.5, 1.5)
        E_next = max(1e-9, E + self.pp.Ts * (self.d - (p_next[0] + p_next[2]) / self.pp.eta_conv))
        I_next = np.array([np.hypot(p_next[0], p_next[1]), np.hypot(p_next[2], p_next[3])]) / np.maximum(V_next, 0.2)
        return p_next, V_next, E_next, I_next


def _tight_limits(cons: ConstraintParams, tg: TighteningParams):
    return dict(Vlo=cons.Vac_min + tg.delta_V, Vhi=cons.Vac_max - tg.delta_V,
                Dlo=cons.Vdc_min + tg.delta_dc, Dhi=cons.Vdc_max - tg.delta_dc,
                Imax=cons.I_max - tg.delta_I)


# =============================================================================
# A. Simplex-style runtime assurance
# =============================================================================
class SimplexRTASupervisor:
    """Decision module of a Simplex architecture (advanced controller =
    candidate OFO, baseline/safety controller = the existing droop fallback).

    Safety monitor (recoverability check), evaluated at every step k:
      predict with the nominal model the trajectory obtained by applying the
      CANDIDATE command u_k^cand for one step and the FALLBACK law
      u = pi_safe(y_hat) for the remaining N_p - 1 steps; the check passes
      iff every predicted V1, V2, Vdc, I1, I2 over the N_p steps lies inside
      the tightened operational set.
    Switching:  s_k = 0 as soon as the check fails while s_{k-1}=1.
    Return:     s_k = 1 only after M_ret consecutive passing checks.
    Initial:    s = 0 (fallback) until the predictor is initialized and the
                dwell condition is met, mirroring CRI's start in fallback.
    The monitor never sees CRI indicators; it is purely model-predictive.
    """
    name = "RTA"

    def __init__(self, pp: PlantParams, cp: ControllerParams, cons: ConstraintParams,
                 prm: Optional[RTAParams] = None, pred_prm: Optional[PredictorParams] = None):
        self.pp, self.cons = pp, cons
        self.prm = prm or RTAParams()
        self.pred = NominalMicrogridPredictor(pp, pred_prm)
        self.fallback_model = FallbackDroopController(cp, cons, pp)   # stateless law, used inside the prediction
        self.lim = _tight_limits(cons, self.prm.tightening)
        self.reset()

    def reset(self):
        self.pred.reset()
        self.s_prev = 0
        self.pass_count = 0
        self.u_prev = None

    def _check(self, u_cand):
        lim = self.lim
        p, V, E, _ = self.pred.current_state()
        min_margin = np.inf
        u = u_cand
        for j in range(self.prm.horizon_steps):
            p, V, E, I = self.pred.step_model(p, V, E, u)
            Vdc = self.pred.vdc_of(E)
            m = min(V.min() - lim["Vlo"], lim["Vhi"] - V.max(), Vdc - lim["Dlo"], lim["Dhi"] - Vdc,
                    lim["Imax"] - I.max())
            min_margin = min(min_margin, m)
            if m < 0:
                return False, min_margin, j + 1
            y_hat = np.array([V[0], V[1], Vdc, I[0], I[1], 0.0, 0.0])
            u = self.fallback_model.command(y_hat)
        return True, min_margin, None

    def step(self, k, t, y_meas, u_applied, u_safe, u_candidate) -> Dict:
        t0 = time.perf_counter_ns()
        self.pred.init_actuator_state(u_applied)
        self.pred.observe(y_meas, None if k == 0 else u_applied)
        if not self.pred.ready:
            ok, mm, j_fail = False, np.nan, None
        else:
            ok, mm, j_fail = self._check(u_candidate)
        if ok:
            self.pass_count += 1
        else:
            self.pass_count = 0
        if self.s_prev == 1:
            s = 1 if ok else 0
        else:
            s = 1 if self.pass_count >= self.prm.return_dwell_steps else 0
        self.s_prev = s
        return dict(s=s, kappa=1.0, cls=("Activate" if s else "Reject"), u_override=None,
                    monitor_pass=int(ok), monitor_min_margin=mm,
                    monitor_fail_step=(np.nan if j_fail is None else j_fail),
                    sup_time_s=(time.perf_counter_ns() - t0) / 1e9)


# =============================================================================
# B. Discrete-time CBF safety filter (CBF-QP, nominal model)
# =============================================================================
class DiscreteCBFFilter:
    """Minimally invasive safety filter on the candidate command.

    Barriers (tightened limits, superscript t):
      DC link (relative degree 1, written in stored energy E = Cdc Vdc^2/2):
        h_dc^lo = E - E(Vdc_min^t),  h_dc^hi = E(Vdc_max^t) - E          (affine in u_k)
      AC voltage (relative degree 2: V_{k+1} does not depend on u_k), applied to
      the one-step-shifted barrier h~_k := h(V_{k+1}) (discrete high-order CBF):
        h_V^lo,i = V_i - V_min^t,  h_V^hi,i = V_max^t - V_i               (affine in u_k)
      Feeder current (relative degree 1):
        h_I,i = I_max^t - I_i   -> second-order-cone constraint in u_k

    CBF-QP (solved every step with SLSQP; all constraints convex; DC rows
    scaled to voltage-equivalent units so that every row is in pu volts/amps):
      u_k* = argmin_u || D_u^{-1}(u - u_k^cand) ||^2
        s.t. h(x_{k+1}(u)) >= (1 - gamma) h(x_k)                (each barrier)
             u_min <= u <= u_max
    If infeasible, the minimal uniform relaxation t* of all barrier rows is
    computed first and the QP is re-solved with every row relaxed by t*
    (lexicographic soft-constraint handling; no tuned penalty weight).
    t* > 0 is recorded as a CBF-infeasibility event.
    The candidate is nominally always 'active' (s_k = 1), but the applied
    command is u_k*; availability of the UNMODIFIED candidate command is
    reported separately.
    """
    name = "CBF"

    def __init__(self, pp: PlantParams, cons: ConstraintParams,
                 prm: Optional[CBFParams] = None, pred_prm: Optional[PredictorParams] = None):
        self.pp, self.cons = pp, cons
        self.prm = prm or CBFParams()
        self.pred = NominalMicrogridPredictor(pp, pred_prm)
        self.lim = _tight_limits(cons, self.prm.tightening)
        self.u_lo = np.array([cons.P_min, cons.Q_min, cons.P_min, cons.Q_min])
        self.u_hi = np.array([cons.P_max, cons.Q_max, cons.P_max, cons.Q_max])
        self.Du = np.asarray(DEFAULT_D_u, dtype=float)
        self.reset()

    def reset(self):
        self.pred.reset()

    # constraint residuals g(u) >= -sigma (vector of 8) and their Jacobian wrt u
    def _build(self):
        pr, lim, g = self.pred, self.lim, self.prm.gamma
        p, V, E, I_now = pr.current_state()
        A = pr.A
        Ts, eta = self.pp.Ts, self.pp.eta_conv
        Elo, Ehi = pr.E_of(lim["Dlo"]), pr.E_of(lim["Dhi"])
        V1 = np.clip(V + pr.av * (pr.w + pr.M @ p - V), 0.5, 1.5)   # V_{k+1}: independent of u_k
        Ibar = lim["Imax"] - (1 - g) * (lim["Imax"] - I_now)          # required I_{k+1} upper bound

        cE = 1.0 / (self.pp.Cdc * self.pp.Vdc_nom)

        def pnext(u):
            # NOTE: the converter apparent-power saturation |S_i|<=S_rated is NOT modeled
            # inside the QP (assumed inactive) and is not checked afterwards; the plant
            # saturates the applied command if it is exceeded (observed only in S7/S8).
            return p + A * (u - p)

        def resid(u):
            pn = pnext(u)
            En = E + Ts * (pr.d - (pn[0] + pn[2]) / eta)
            V2 = V1 + pr.av * (pr.w + pr.M @ pn - V1)
            r = [
                cE * ((En - Elo) - (1 - g) * (E - Elo)),
                cE * ((Ehi - En) - (1 - g) * (Ehi - E)),
                (V2[0] - lim["Vlo"]) - (1 - g) * (V1[0] - lim["Vlo"]),
                (V2[1] - lim["Vlo"]) - (1 - g) * (V1[1] - lim["Vlo"]),
                (lim["Vhi"] - V2[0]) - (1 - g) * (lim["Vhi"] - V1[0]),
                (lim["Vhi"] - V2[1]) - (1 - g) * (lim["Vhi"] - V1[1]),
                Ibar[0] * V1[0] - np.hypot(pn[0], pn[1]),
                Ibar[1] * V1[1] - np.hypot(pn[2], pn[3]),
            ]
            return np.array(r)

        # DC rows are expressed in voltage-equivalent units (divided by
        # dE/dVdc = Cdc*Vdc_nom) so that every row has a comparable scale.
        dE = -Ts / eta * np.array([A[0], 0.0, A[2], 0.0])
        dV = pr.av * pr.M * A[None, :]      # dV2/du (2x4)

        def jac(u):
            pn = pnext(u)
            J = np.zeros((8, 4))
            J[0], J[1] = cE * dE, -cE * dE
            J[2], J[3] = dV[0], dV[1]
            J[4], J[5] = -dV[0], -dV[1]
            s1, s2 = max(np.hypot(pn[0], pn[1]), 1e-9), max(np.hypot(pn[2], pn[3]), 1e-9)
            J[6] = -np.array([pn[0] * A[0] / s1, pn[1] * A[1] / s1, 0, 0])
            J[7] = -np.array([0, 0, pn[2] * A[2] / s2, pn[3] * A[3] / s2])
            return J
        return resid, jac

    def _solve(self, resid, jac, uc):
        """Lexicographic CBF-QP.
        Stage A: hard-constrained QP  min ||D_u^-1(u-uc)||^2  s.t. r(u) >= 0, box.
        If A is infeasible (SLSQP failure or residual < -feas_tol):
        Stage B: minimal uniform relaxation  t* = min t  s.t. r(u) + t >= 0, t >= 0, box;
        Stage C: QP of Stage A with r(u) + t* >= 0.
        Returns (u*, t*, status, iterations); status 0 = A solved (t*=0),
        1 = relaxed (B+C solved), >=10 = solver failure in the indicated stage."""
        W = 1.0 / self.Du ** 2
        bnds = [(lo, hi) for lo, hi in zip(self.u_lo, self.u_hi)]
        opts = {"maxiter": self.prm.maxiter, "ftol": self.prm.ftol}
        tol = self.prm.feas_tol
        f = lambda u: float(np.sum(W * (u - uc) ** 2))
        fg = lambda u: 2 * W * (u - uc)
        rA = minimize(f, uc, jac=fg, method="SLSQP", bounds=bnds,
                      constraints=[{"type": "ineq", "fun": resid, "jac": jac}], options=opts)
        uA = np.clip(rA.x, self.u_lo, self.u_hi)
        if rA.success and resid(uA).min() >= -tol:
            return uA, 0.0, 0, int(rA.nit)
        # Stage B
        eps = 1e-6
        fB = lambda x: float(x[4] + eps * np.sum(W * (x[:4] - uc) ** 2))
        fBg = lambda x: np.r_[2 * eps * W * (x[:4] - uc), 1.0]
        cB = lambda x: resid(x[:4]) + x[4]
        cBj = lambda x: np.hstack([jac(x[:4]), np.ones((8, 1))])
        t0 = max(0.0, -resid(uc).min())
        rB = minimize(fB, np.r_[uc, t0], jac=fBg, method="SLSQP", bounds=bnds + [(0.0, None)],
                      constraints=[{"type": "ineq", "fun": cB, "jac": cBj}], options=opts)
        t_star = max(0.0, float(rB.x[4]))
        if not rB.success or cB(rB.x).min() < -tol:
            return np.clip(rB.x[:4], self.u_lo, self.u_hi), t_star, 20 + int(rB.status), int(rA.nit + rB.nit)
        # Stage C
        cC = lambda u: resid(u) + t_star + tol
        rC = minimize(f, np.clip(rB.x[:4], self.u_lo, self.u_hi), jac=fg, method="SLSQP", bounds=bnds,
                      constraints=[{"type": "ineq", "fun": cC, "jac": jac}], options=opts)
        uC = np.clip(rC.x, self.u_lo, self.u_hi)
        if not rC.success or cC(uC).min() < -tol:
            uB = np.clip(rB.x[:4], self.u_lo, self.u_hi)      # B's point is feasible for the relaxed set
            return uB, t_star, 30 + int(rC.status), int(rA.nit + rB.nit + rC.nit)
        return uC, t_star, 1, int(rA.nit + rB.nit + rC.nit)

    def step(self, k, t, y_meas, u_applied, u_safe, u_candidate) -> Dict:
        t0 = time.perf_counter_ns()
        self.pred.init_actuator_state(u_applied)
        self.pred.observe(y_meas, None if k == 0 else u_applied)
        uc = np.clip(np.asarray(u_candidate, dtype=float), self.u_lo, self.u_hi)
        info = dict(s=1, kappa=1.0, cls="Activate", cbf_status=-1, cbf_slack=0.0,
                    cbf_nit=0, cbf_active=0, cbf_resid_min=np.nan)
        if not self.pred.ready:
            info.update(u_override=uc, cbf_status=-2)     # no model yet: pass-through (k=0 only)
        else:
            resid, jac = self._build()
            r0 = resid(uc)
            info["cbf_resid_min"] = float(r0.min())
            if np.all(r0 >= 0.0):
                info.update(u_override=uc, cbf_status=0)  # candidate already satisfies every CBF condition
            else:
                u_star, t_star, status, nit = self._solve(resid, jac, uc)
                info.update(u_override=u_star, cbf_status=status, cbf_slack=t_star, cbf_nit=nit,
                            cbf_active=int(np.sum(resid(u_star) + t_star < 1e-7)),
                            cbf_resid_after=float(resid(u_star).min()))
        info["filtered"] = int(np.max(np.abs(info["u_override"] - np.asarray(u_candidate))) > self.prm.modified_tol)
        info["filter_dev_inf"] = float(np.max(np.abs(info["u_override"] - np.asarray(u_candidate))))
        info["sup_time_s"] = (time.perf_counter_ns() - t0) / 1e9
        return info
