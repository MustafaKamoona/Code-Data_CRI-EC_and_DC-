"""Measurement Validity Gate (MVG).

A lightweight preprocessing/supervisory gate placed in front of
the unchanged CRISupervisor. It does not modify the five readiness
indicators, their equations, the calibration procedure, or any control law.

Per-sample checks on the DELIVERED measurement y_k (and its timestamp tau_k
when available):
  (1) finite      : every channel is finite (rejects NaN/Inf, dropped packets);
  (2) fresh       : if timestamps are available, t_k - tau_k <= tau_max and
                    tau_k advances strictly (rejects stale/repeated/delayed data);
  (3) plausible   : y_k lies inside a wide physical sensor-range envelope
                    (rejects gross outliers only -- not moderate ones);
  (4) not frozen  : no channel has repeated exactly for L consecutive samples
                    while at least one of the inputs that physically drive that
                    channel was changing at every one of those samples
                    (channel-specific excitation condition; e.g. Vdc is driven
                    only by P1*, P2*, so Q-only excitation does not test it).
                    Current channels reading
                    exactly zero are EXEMPT (a de-energized feeder, e.g. after
                    the S7/S8 VSC2 outage, is legitimately constant); a
                    stuck-at-zero current sensor is therefore NOT detected.

State machine:  VALID --fail--> INVALID --pass--> RECOVERING --R passes--> VALID
                (any failure while RECOVERING returns to INVALID).
Actions requested from the runner (see mvg_sim.py):
  INVALID    : CRI is not updated (the sample never reaches the sensitivity
               estimator or any indicator); candidate authority s_k = 0 and the
               candidate's internal update is frozen; the DT/operator escalation
               flag is raised. The fallback receives, channel by channel, the
               current value if that channel is usable for the fallback (finite,
               plausible, not frozen, packet age <= tau_fb_max) and otherwise the
               last usable value of that channel (held); an invalid value is never
               passed to it.
  RECOVERING : on entry the CRI state is reset (existing CRISupervisor.reset()),
               so no pre-fault data remain in its windows; valid samples feed the
               CRI normally, but s_k is held at 0 and the DT flag stays raised for
               R consecutive valid samples.
A requested fallback transition is NOT a guarantee of safe operation: the
fallback then runs on held (open-loop) data, and its safety still rests on
the assumption that the fallback controller keeps the plant admissible.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List
import numpy as np

from .config import CRIParams

Y_NAMES = ["V1", "V2", "Vdc", "I1", "I2", "P_POI", "Q_POI"]
VALID, INVALID, RECOVERING = "VALID", "INVALID", "RECOVERING"


@dataclass
class MVGParams:
    # (3) wide sensor-range envelope [pu], ~2x around the operating region and >= 10 sigma
    # of the largest simulated noise (2 %); the plant itself clips V, Vdc to [0.5, 1.5].
    y_lo: Tuple[float, ...] = (0.3, 0.3, 0.3, -0.2, -0.2, -5.0, -5.0)
    y_hi: Tuple[float, ...] = (1.7, 1.7, 1.7, 3.0, 3.0, 5.0, 5.0)
    # (2) staleness: maximum admissible measurement age [s]; None disables the check
    tau_max: Optional[float] = 0.03            # 1.5 Ts at Ts = 0.02 s
    # (4) frozen-channel test
    frozen_L: int = 10                         # consecutive exact repeats (0.2 s)
    frozen_tol: float = 0.0                    # exact repetition (stuck digital value)
    u_excitation_min: float = 1e-9             # |Delta u|_inf required at every one of those samples
    zero_exempt_channels: Tuple[int, ...] = (3, 4)   # I1, I2: exactly-zero current is legitimate
    zero_tol: float = 1e-9
    # inputs [P1*, Q1*, P2*, Q2*] that physically drive each channel (plant structure of plant.HybridACDCPlant)
    channel_inputs: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 3), (0, 1, 2, 3), (0, 2), (0, 1), (2, 3), (0, 2), (1, 3))
    # fallback input: maximum packet age [s] for a channel value still used by the fallback
    tau_fb_max: Optional[float] = 0.2
    # recovery: R consecutive valid samples; None -> max(N, 2*Na) from CRIParams, i.e. every
    # CRI data window (sensitivity window N, adaptability window 2*Na) holds post-fault data only
    recovery_samples: Optional[int] = None
    # apply the recovery requirement also at cold start (changes startup behaviour; default off
    # so that fault-free operation is identical to the published implementation)
    startup_requires_recovery: bool = False


class MeasurementValidityGate:
    def __init__(self, prm: Optional[MVGParams] = None, crip: Optional[CRIParams] = None):
        self.prm = prm or MVGParams()
        crip = crip or CRIParams()
        self.R = self.prm.recovery_samples if self.prm.recovery_samples is not None else max(crip.N, 2 * crip.Na)
        self.lo, self.hi = np.asarray(self.prm.y_lo, float), np.asarray(self.prm.y_hi, float)
        self.reset()

    def reset(self):
        self.state = RECOVERING if self.prm.startup_requires_recovery else VALID
        self.recover_count = 0
        self.y_hist: List[np.ndarray] = []      # recent delivered measurements (finite ones)
        self.du_hist: List[np.ndarray] = []     # recent |Delta u| per input of the applied command
        self.u_prev = None
        self.ts_prev = None
        self.frozen_flag = np.zeros(len(self.lo), dtype=bool)
        self.n_invalid = 0

    def _frozen(self, y):
        L, tol = self.prm.frozen_L, self.prm.frozen_tol
        flag = self.frozen_flag.copy()
        if len(self.y_hist) >= L and len(self.du_hist) >= L:
            D = np.stack(self.du_hist[-L:])                            # (L, n_u)
            excited = np.array([bool(np.all(D[:, list(ins)].max(axis=1) > self.prm.u_excitation_min))
                                for ins in self.prm.channel_inputs])
            H = np.stack(self.y_hist[-L:] + [y])                       # (L+1, n_y)
            rep = np.all(np.abs(np.diff(H, axis=0)) <= tol, axis=0)    # unchanged over L steps
            new = rep & excited
            for j in self.prm.zero_exempt_channels:
                if abs(y[j]) <= self.prm.zero_tol:
                    new[j] = False
            flag |= new
        if len(self.y_hist):
            changed = np.abs(y - self.y_hist[-1]) > tol
            flag &= ~changed                                           # a flagged channel clears once it moves
        for j in self.prm.zero_exempt_channels:
            if abs(y[j]) <= self.prm.zero_tol:
                flag[j] = False
        return flag

    def check(self, t: float, y: np.ndarray, ts: Optional[float], u_applied: np.ndarray):
        """Evaluate sample k. Returns dict(valid, state, reasons, reset_cri, recovering)."""
        y = np.asarray(y, dtype=float)
        u_applied = np.asarray(u_applied, dtype=float)
        if self.u_prev is not None:
            self.du_hist.append(np.abs(u_applied - self.u_prev))
            self.du_hist = self.du_hist[-(self.prm.frozen_L + 1):]
        self.u_prev = u_applied.copy()

        reasons = []
        fin_mask = np.isfinite(y)
        finite = bool(np.all(fin_mask))
        if not finite:
            reasons.append("nonfinite")
        if self.prm.tau_max is not None and ts is not None:
            if (t - ts) > self.prm.tau_max + 1e-12 or (self.ts_prev is not None and ts <= self.ts_prev):
                reasons.append("stale")
        with np.errstate(invalid="ignore"):
            plaus_mask = fin_mask & (y >= self.lo) & (y <= self.hi)
        if finite and not np.all(plaus_mask):
            reasons.append("implausible")
        fz = np.zeros(len(y), dtype=bool)
        if finite:
            fz = self._frozen(y)
            self.frozen_flag = fz
            if fz.any():
                reasons.append("frozen:" + ",".join(Y_NAMES[j] for j in np.where(fz)[0]))
            self.y_hist.append(y.copy())
            self.y_hist = self.y_hist[-(self.prm.frozen_L + 1):]
        if ts is not None and "stale" not in reasons:
            self.ts_prev = ts

        # channel usability for the FALLBACK (less strict than validity for the CRI/candidate)
        fresh_fb = True
        if self.prm.tau_fb_max is not None and ts is not None:
            fresh_fb = (t - ts) <= self.prm.tau_fb_max + 1e-12
        chan_ok_fb = plaus_mask & ~fz & fresh_fb

        ok = not reasons
        reset_cri = False
        if not ok:
            self.state, self.recover_count = INVALID, 0
            self.n_invalid += 1
        elif self.state == INVALID:
            self.state, self.recover_count, reset_cri = RECOVERING, 1, True
        elif self.state == RECOVERING:
            self.recover_count += 1
        if self.state == RECOVERING and self.recover_count >= self.R:
            self.state = VALID
        return dict(valid=ok, state=self.state, reasons=";".join(reasons), reset_cri=reset_cri,
                    recovering=(self.state == RECOVERING), chan_ok_fb=chan_ok_fb)
