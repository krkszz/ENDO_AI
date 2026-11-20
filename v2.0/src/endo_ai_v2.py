"""
ENDO_AI v2.0 -- perfect-foresight general equilibrium with AI
=============================================================

Same technology as v1.0 (task-based production, automated research, recursive
AI-software progress, compute and power constraints), but the rule-based
closure is replaced by forward-looking agents:

* Households (Ramsey, CRRA) choose consumption; the interest rate path r_t
  satisfies the Euler equation.
* Physical capital is demanded by goods firms until MP_K - delta_K = r.
* Compute owners invest in chips + power until the rental rate of compute
  covers the user cost, which includes interest, depreciation AND the expected
  capital loss from falling chip prices (no-arbitrage with perfect foresight).
* AI labs own the software S, sell AI services at a markup mu over the
  compute cost, and choose the share h_t of their compute devoted to R&D
  (training, AI-software research, experiments) so that the present value of
  the future rents it creates equals its opportunity cost.

The equilibrium is computed with nested loops:
  inner : Newton's method on the interest-rate path r_t so that the Euler
          equation holds (Jacobian by vectorised finite differences)
  outer : the lab R&D policy h_t (piecewise linear in logit space between
          11 knots) maximises the PV of lab profits given the price path;
          the outer fixed point between prices and policy is damped.

All arrays carry a leading "draw" dimension N so several parameter sets
(scenarios, Monte Carlo draws) are solved simultaneously.
Time is annual: t = 0 is 2025, horizon T years (default 100, results are
reported to 2075 so the terminal condition has little influence).
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    T=100,               # horizon in years (2025 .. 2125)
    # preferences
    theta=1.5,           # CRRA / inverse elasticity of intertemporal substitution
    rho_disc=0.03,       # pure rate of time preference
    # goods production
    alpha=0.35, eps=0.6, delta_K=0.05, gK_max=0.30, n=0.004,
    # automation
    beta0=0.003, beta_max=0.95, betaR0=0.003, betaR_max=0.95,
    kappa_mid=12.0, kappaR_extra=3.0, tau=8.0, tau_R=3.0, gate_slope=2.0,
    # compute & power
    delta_H=0.20, g_Q=0.25, wright_b=0.10,
    g_eta=0.12, g_P=0.15, power_cost=0.5, delta_P=0.04, E0_mult=2.0,
    u_RA=0.10,           # share of compute for general (public) research, fixed
    f_T=0.6, f_RS=0.3, f_exp=0.1,   # split of lab R&D compute
    h0=0.5,              # initial guess for lab R&D share of compute
    mu=2.0,              # markup of AI labs on AI services (gross margin 50%)
    c_R=30.0,
    # TFP research
    th_A=0.012, lam_A=0.5, phi_A=0.5, LA_share=0.015,
    # AI software research
    th_S=0.9, lam_S=0.6, r_S=1.0, gam=0.3, lnS_max=np.log(1e6),
    LS_share=0.0005, g_LS=0.08, LS_mult_max=5.0,
    # 2025 levels
    Y0=100.0, KY0=3.0, r0=0.05,
)

SCENARIOS = {
    "Bottleneck": dict(eps=0.4, r_S=0.7, beta_max=0.70, betaR_max=0.60,
                       kappa_mid=16.0, tau=12.0, g_P=0.08),
    "Transformative": dict(eps=0.6, r_S=1.0, beta_max=0.95, betaR_max=0.95,
                           kappa_mid=12.0, tau=7.0, g_P=0.15),
    "Explosive": dict(eps=0.8, r_S=1.4, beta_max=0.995, betaR_max=0.995,
                      kappa_mid=9.0, tau=4.0, g_P=0.30),
}

X_MIN, X_MAX = np.log(0.97), np.log(60.0)   # bounds on log(1+r) during iteration


def make_params(N=1, **kw):
    p = {**DEFAULTS, **kw}
    out = {}
    for k, v in p.items():
        if k == "T":
            out[k] = int(v)
        else:
            out[k] = np.broadcast_to(np.asarray(v, float), (N,)).copy()
    out["N"] = N
    return out


def tile_params(P, reps):
    """Repeat every draw `reps` times (draw-major) for perturbation runs."""
    Q = {k: (np.repeat(v, reps) if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    Q["N"] = P["N"] * reps
    return Q


# ----------------------------------------------------------------------------
# Technology blocks
# ----------------------------------------------------------------------------
def ces(xa, xl, b, eps):
    rho = (eps - 1.0) / eps
    B = b ** (1.0 / eps) * xa ** rho + (1.0 - b) ** (1.0 / eps) * xl ** rho
    return B ** (1.0 / rho), B, rho


def goods(A, K, X, LY, bt, alpha, eps):
    Z, B, rho = ces(X, LY, bt, eps)
    Y = A * K ** alpha * Z ** (1 - alpha)
    w = (1 - alpha) * Y * (1 - bt) ** (1 / eps) * LY ** (rho - 1) / B
    MPX = (1 - alpha) * Y * bt ** (1 / eps) * X ** (rho - 1) / B
    return Y, Z, w, MPX


SHARP = 25.0     # sharpness of smooth bounds (in log units)


def smooth_clip_log(x, lo, hi, k=SHARP):
    """Smoothly clip x into [lo, hi] (all positive) in log space."""
    lx, llo, lhi = np.log(x), np.log(lo), np.log(hi)
    y = llo + np.logaddexp(0.0, k * (lx - llo)) / k            # smooth max(lx, llo)
    y = lhi - np.logaddexp(0.0, k * (lhi - y)) / k             # smooth min(y, lhi)
    return np.exp(y)


def logistic(k, mid, width, top):
    return top / (1.0 + np.exp(-(k - mid) / width))


# ----------------------------------------------------------------------------
# Calibration of 2025
# ----------------------------------------------------------------------------
def calibrate(P):
    """Pin down c (compute per AI worker-equivalent) so that in 2025 the
    rental value of compute equals its user cost (no-arbitrage), then A0 so
    that Y0 = 100."""
    N = P["N"]
    L0 = np.ones(N)
    LY0 = L0 * (1 - P["LA_share"] - P["LS_share"])
    K0 = P["KY0"] * P["Y0"]
    H0 = np.ones(N)
    Q0 = np.ones(N)                       # FLOP/s per $ (chips), normalised
    eta0 = np.ones(N)
    pP = P["power_cost"] / Q0             # $ per unit of power
    pi0 = 1 / Q0 + pP / eta0
    UC0 = ((1 + P["r0"]) * pi0 - (1 - P["delta_H"]) / (Q0 * np.exp(P["g_Q"]))
           - (1 - P["delta_P"]) * pP / eta0)
    u_inf = 1 - P["u_RA"] - P["h0"]
    bt0 = P["beta0"]

    def excess(lnc):
        c = np.exp(lnc)
        X = H0 * u_inf / c
        Z, _, _ = ces(X, LY0, bt0, P["eps"])
        A = P["Y0"] / (K0 ** P["alpha"] * Z ** (1 - P["alpha"]))
        Y, _, w, MPX = goods(A, K0, X, LY0, bt0, P["alpha"], P["eps"])
        R_H = MPX / (c * P["mu"])
        return np.log(R_H) - np.log(UC0), A

    lo, hi = np.full(N, -5.0), np.full(N, 15.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f, _ = excess(mid)
        # R_H is proportional to the AI share of the task bundle, which rises
        # with c when eps < 1 (AI scarcer) and falls when eps > 1
        up = (f > 0) == (P["eps"] < 1)
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    lnc = 0.5 * (lo + hi)
    f, A0 = excess(lnc)
    c = np.exp(lnc)

    S0 = np.ones(N)
    h = P["h0"]
    XRA0 = S0 * H0 * P["u_RA"] / (c * P["c_R"])
    XRS0 = S0 * H0 * h * P["f_RS"] / (c * P["c_R"])
    RA0, _, _ = ces(XRA0, P["LA_share"] * L0, P["betaR0"], P["eps"])
    RS0, _, _ = ces(XRS0, P["LS_share"] * L0, P["betaR0"], P["eps"])
    cal = dict(c=c, A0=A0, RA0=RA0, RS0=RS0, Hexp0=H0 * h * P["f_exp"],
               CT_norm=3.0 * S0 * H0 * h * P["f_T"], pP=pP,
               K0=K0 * np.ones(N), H0=H0, Q0=Q0, eta0=eta0,
               P0=P["E0_mult"] * H0, calib_residual=np.abs(f).max())
    return cal


# ----------------------------------------------------------------------------
# Forward simulation given price path r and lab policy h
# ----------------------------------------------------------------------------
def forward(P, cal, r, h, Qpath=None, n_bis=44):
    """Simulate t = 0..T given interest rates r[:, t] (return from t-1 to t,
    t = 1..T+1) and lab R&D shares h[:, t].  Returns a dict of (N, T+1)
    arrays.  Qpath (N, T+2), if given, is the expected chip price-performance
    path used in the user cost (perfect foresight on prices)."""
    T, N = P["T"], P["N"]
    al, ep = P["alpha"], P["eps"]
    phi_S = 1.0 - P["lam_S"] / P["r_S"]
    w_Y = P["kappa_mid"] / np.log(P["beta_max"] / P["beta0"] - 1.0)
    midR = P["kappa_mid"] + P["kappaR_extra"]
    w_R = midR / np.log(P["betaR_max"] / P["betaR0"] - 1.0)
    c, A0 = cal["c"], cal["A0"]
    pP = cal["pP"]

    K = cal["K0"].copy(); H = cal["H0"].copy(); Pw = cal["P0"].copy()
    Q = cal["Q0"].copy(); eta = cal["eta0"].copy()
    lnS = np.zeros(N); a = np.ones(N)
    bt = P["beta0"].copy(); btR = P["betaR0"].copy()
    cumC = np.ones(N); cumI = 3.0 * np.ones(N); L = np.ones(N)

    keys = ["Y", "C", "c_pc", "w", "MPX", "MPK", "R_H", "UC", "lab_profit",
            "I_K", "I_chip", "I_P", "AI_capex_share", "save_rate", "beta",
            "betaR", "lnS", "gS", "gA", "H", "K", "power_bind", "h", "Q",
            "L", "lab_share", "price_AI", "X"]
    o = {k: np.zeros((N, T + 1)) for k in keys}
    Qnext_used = np.zeros((N, T + 2)); Qnext_used[:, 0] = Q

    for t in range(T + 1):
        S = np.exp(lnS)
        A = A0 * a
        LY = L * (1 - P["LA_share"] - P["LS_share"])
        ht = h[:, t]
        u_inf = 1 - P["u_RA"] - ht
        X = S * u_inf * H / c
        Y, Z, w, MPX = goods(A, K, X, LY, bt, al, ep)
        MPK = al * Y / K
        R_H = MPX * S / (c * P["mu"])         # rental of compute (lab FOC on inference)
        price_AI = P["mu"] * R_H * c / S       # price of one AI worker-equivalent
        lab_profit = MPX * X - R_H * (1 - P["u_RA"]) * H

        # ---- research & capability (flows at t determine states at t+1) ----
        LS = P["LS_share"] * L * np.minimum(np.exp(P["g_LS"] * t), P["LS_mult_max"])
        LA = P["LA_share"] * L
        XRA = S * H * P["u_RA"] / (c * P["c_R"])
        XRS = S * H * ht * P["f_RS"] / (c * P["c_R"])
        RA, _, _ = ces(XRA, LA, btR, ep)
        RS, _, _ = ces(XRS, LS, btR, ep)
        gA = P["th_A"] * a ** (P["phi_A"] - 1) * (RA / cal["RA0"]) ** P["lam_A"]
        ceil = np.clip(1.0 - lnS / P["lnS_max"], 0.0, 1.0)
        gS = (P["th_S"] * S ** (phi_S - 1) * (RS / cal["RS0"]) ** P["lam_S"]
              * (H * ht * P["f_exp"] / cal["Hexp0"]) ** P["gam"] * ceil)
        gS = np.minimum(gS, 5.0)
        CT = S * H * ht * P["f_T"]
        cumC_n = cumC + CT / cal["CT_norm"]
        kappa = np.log(cumC_n)
        beta_cap = logistic(kappa, P["kappa_mid"], w_Y, P["beta_max"])
        betaR_cap = logistic(kappa, midR, w_R, P["betaR_max"])
        gate = 1.0 / (1.0 + np.exp(-P["gate_slope"] * np.log(w / price_AI)))
        bt_n = np.clip(bt + (np.maximum(beta_cap * gate, P["beta0"]) - bt) / P["tau"], 1e-6, 1 - 1e-6)
        btR_n = np.clip(btR + (betaR_cap - btR) / P["tau_R"], 1e-6, 1 - 1e-6)
        a_n = a * np.exp(gA)
        lnS_n = np.minimum(lnS + gS, P["lnS_max"])
        L_n = L * np.exp(P["n"])
        eta_n = eta * np.exp(P["g_eta"])
        S_n, A_n = np.exp(lnS_n), A0 * a_n
        LY_n = L_n * (1 - P["LA_share"] - P["LS_share"])

        # ---- investment decisions, perfect foresight ----
        r_n = r[:, min(t + 1, T + 1)]
        Q_n = Qpath[:, t + 1] if Qpath is not None else Q * np.exp(P["g_Q"])
        pi_t = 1 / Q + pP / eta
        UC = (1 + r_n) * pi_t - (1 - P["delta_H"]) / Q_n - (1 - P["delta_P"]) * pP / eta
        UC = np.maximum(UC, 1e-6 * pi_t)
        h_n = h[:, min(t + 1, T)]
        u_inf_n = 1 - P["u_RA"] - h_n
        K_lo = (1 - P["delta_K"]) * K
        K_hi = K * np.exp(P["gK_max"])

        def K_of(Hn):
            Xn = S_n * u_inf_n * Hn / c
            Zn, _, _ = ces(Xn, LY_n, bt_n, ep)
            Kstar = (al * A_n * Zn ** (1 - al) / (r_n + P["delta_K"])) ** (1 / (1 - al))
            return smooth_clip_log(Kstar, K_lo, K_hi), Xn

        def excess(Hn):
            Kn, Xn = K_of(Hn)
            _, _, _, MPXn = goods(A_n, Kn, Xn, LY_n, bt_n, al, ep)
            return np.log(MPXn * S_n / (c * P["mu"])) - np.log(UC)

        # unconstrained compute demand H* (no-arbitrage), then smooth bounds:
        # irreversibility below, power build-out limit above
        H_lo = (1 - P["delta_H"]) * H
        P_max = Pw * (1 - P["delta_P"]) * np.exp(P["g_P"])
        H_cap = np.maximum(P_max * eta_n, H_lo * 1.0001)
        lo, hi = np.log(H_lo) - 6.0, np.log(H_cap) + 6.0
        for _ in range(n_bis):
            mid = 0.5 * (lo + hi)
            f = excess(np.exp(mid))
            lo = np.where(f > 0, mid, lo)
            hi = np.where(f > 0, hi, mid)
        H_star = np.exp(0.5 * (lo + hi))
        H_n = smooth_clip_log(H_star, H_lo, H_cap)
        bind = H_star > H_cap
        K_n, _ = K_of(H_n)
        P_n = np.maximum(H_n / eta_n, Pw * (1 - P["delta_P"]))

        I_chip = np.maximum(H_n - (1 - P["delta_H"]) * H, 0.0) / Q
        I_P = np.maximum(P_n - (1 - P["delta_P"]) * Pw, 0.0) * pP
        I_K = K_n - (1 - P["delta_K"]) * K
        C = Y - I_K - I_chip - I_P

        # ---- record ----
        for k, v in (("Y", Y), ("C", C), ("c_pc", C / L), ("w", w), ("MPX", MPX),
                     ("MPK", MPK), ("R_H", R_H), ("UC", UC), ("lab_profit", lab_profit),
                     ("I_K", I_K), ("I_chip", I_chip), ("I_P", I_P),
                     ("AI_capex_share", (I_chip + I_P) / Y), ("save_rate", 1 - C / Y),
                     ("beta", bt), ("betaR", btR), ("lnS", lnS), ("gS", gS), ("gA", gA),
                     ("H", H), ("K", K), ("power_bind", bind.astype(float)), ("h", ht),
                     ("Q", Q), ("L", L), ("lab_share", w * L / Y), ("price_AI", price_AI),
                     ("X", X)):
            o[k][:, t] = v

        # ---- advance ----
        cumI = cumI + I_chip
        Q = Q * np.exp(P["g_Q"]) * (1 + I_chip / cumI) ** P["wright_b"]
        Qnext_used[:, t + 1] = Q
        K, H, Pw = K_n, H_n, P_n
        lnS, a, bt, btR, cumC = lnS_n, a_n, bt_n, btR_n, cumC_n
        L, eta = L_n, eta_n

    o["Qpath"] = Qnext_used
    o["g"] = np.concatenate([np.full((N, 1), np.nan), np.diff(np.log(o["Y"]), axis=1)], axis=1)
    return o


# ----------------------------------------------------------------------------
# Inner loop: interest rates from the Euler equation
# ----------------------------------------------------------------------------
def euler_rates(P, o):
    """Interest rate r_{t+1} implied by the Euler equation given the c_pc
    path.  Consumption is floored smoothly at 2% of output per head so that
    the map stays differentiable when a guess implies negative consumption."""
    beta = 1.0 / (1.0 + P["rho_disc"])
    y = o["Y"] / o["L"]
    fl = 0.02 * y
    c = np.where(o["c_pc"] > fl, o["c_pc"], fl * np.exp((o["c_pc"] - fl) / fl))
    gc = c[:, 1:] / c[:, :-1]
    return gc ** P["theta"][:, None] / beta[:, None] - 1.0


def _residual(P, cal, x, h, Qpath):
    """Euler residual F = log(1 + r_Euler) - log(1 + r) for t = 1..T."""
    T, N = P["T"], P["N"]
    r = np.empty((N, T + 2))
    r[:, 0] = 0.0
    r[:, 1:T + 1] = np.expm1(x)
    r[:, T + 1] = r[:, T]
    o = forward(P, cal, r, h, Qpath)
    F = np.log1p(np.maximum(euler_rates(P, o), -0.99)) - x
    return F, o, r


def solve_rates(P, cal, h, r=None, Qpath=None, tol=1e-6, max_it=40, verbose=False):
    """Newton solver for the interest-rate path.  The Jacobian of the Euler
    residuals with respect to log(1+r_t) is computed by forward differences;
    all T perturbed paths of all N draws run as one vectorised simulation."""
    T, N = P["T"], P["N"]
    if r is None:
        x = np.full((N, T), np.log1p(0.05))
    else:
        x = np.log1p(r[:, 1:T + 1])
    F, o, r = _residual(P, cal, x, h, Qpath)
    PT = tile_params(P, T)
    calT = {k: (np.repeat(v, T) if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == N else v)
            for k, v in cal.items()}
    hT = np.repeat(h, T, axis=0)
    QT = None if Qpath is None else np.repeat(Qpath, T, axis=0)
    lam_grid = np.array([1.0, 0.5, 0.25, 0.1, 0.03])
    nl = len(lam_grid)
    PL = tile_params(P, nl)
    calL = {k: (np.repeat(v, nl) if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == N else v)
            for k, v in cal.items()}
    hL = np.repeat(h, nl, axis=0)
    QL = None if Qpath is None else np.repeat(Qpath, nl, axis=0)
    d = 1e-4
    for it in range(max_it):
        err = np.abs(F[:, :T - 10]).max(axis=1)
        if verbose:
            print(f"    newton {it:2d}: max Euler gap {err.max():.2e}")
        if err.max() < tol:
            break
        xT = np.repeat(x, T, axis=0)
        j = np.tile(np.arange(T), N)
        xT[np.arange(N * T), j] += d
        FT, _, _ = _residual(PT, calT, xT, hT, QT)
        J = ((FT.reshape(N, T, T) - F[:, None, :]) / d).transpose(0, 2, 1)   # J[n, i, j] = dF_i/dx_j
        try:
            dx = np.linalg.solve(J, -F[..., None])[..., 0]
        except np.linalg.LinAlgError:
            dx = np.stack([np.linalg.lstsq(J[n], -F[n], rcond=None)[0] for n in range(N)])
        dx = np.where(np.isfinite(dx), dx, 0.0)
        dx = np.clip(dx, -1.0, 1.0)
        # line search over step sizes, all draws at once
        xL = np.repeat(x, nl, axis=0) + np.tile(lam_grid, N)[:, None] * np.repeat(dx, nl, axis=0)
        xL = np.clip(xL, X_MIN, X_MAX)
        FL, _, _ = _residual(PL, calL, xL, hL, QL)
        nrm = np.sqrt((FL[:, :T - 10] ** 2).sum(axis=1)).reshape(N, nl)
        nrm = np.where(np.isfinite(nrm), nrm, np.inf)
        best = nrm.argmin(axis=1)
        x = xL.reshape(N, nl, T)[np.arange(N), best]
        F, o, r = _residual(P, cal, x, h, Qpath)
    o["euler_gap"] = np.abs(F[:, :T - 10]).max(axis=1)
    o["r"] = r[:, :T + 1].copy()
    o["r"][:, 0] = np.nan
    return r, o["Qpath"], o


# ----------------------------------------------------------------------------
# Outer loop: lab R&D policy maximises the PV of lab profits given prices
# ----------------------------------------------------------------------------
H_LO, H_HI = 0.02, 0.85            # bounds on the lab R&D share of compute
KNOTS = np.array([0, 3, 6, 10, 15, 20, 25, 30, 40, 50, 75])   # years after 2025


def h_from_knots(z, T):
    """Lab R&D share path from knot values z (logit scale), linear in logit
    between knots, constant after the last knot."""
    t = np.arange(T + 1)
    zt = np.stack([np.interp(t, KNOTS, zi) for zi in z])
    return H_LO + (H_HI - H_LO) / (1 + np.exp(-zt))


def lab_value(o, r, horizon=None):
    """PV (at 2025) of lab profits up to `horizon` years, and discount factors."""
    T = o["Y"].shape[1] - 1
    hz = T if horizon is None else horizon
    D = np.ones_like(o["Y"])
    D[:, 1:] = np.cumprod(1.0 / (1.0 + r[:, 1:T + 1]), axis=1)
    return (D * o["lab_profit"])[:, :hz + 1].sum(axis=1), D


def _rep(P, cal, r, Qpath, m):
    N = P["N"]
    Pm = tile_params(P, m)
    calm = {k: (np.repeat(v, m) if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == N else v)
            for k, v in cal.items()}
    return Pm, calm, np.repeat(r, m, axis=0), np.repeat(Qpath, m, axis=0)


def lab_optimize(P, cal, r, Qpath, z, n_it=12, dz=0.05, horizon=None, verbose=False):
    """Maximise PV of lab profits over knot values z, holding the price path
    (r, chip prices) fixed: central-difference gradient + vectorised line
    search that only accepts improvements."""
    N, T = P["N"], P["T"]
    hz = horizon or (T - 25)
    nk = len(KNOTS)
    m_g = 2 * nk
    Pg, calg, rg, Qg = _rep(P, cal, r, Qpath, m_g)
    steps = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    m_s = len(steps)
    Ps, cals, rs, Qs = _rep(P, cal, r, Qpath, m_s)
    o = forward(P, cal, r, h_from_knots(z, T), Qpath)
    V, _ = lab_value(o, r, hz)
    for it in range(n_it):
        zz = np.repeat(z, m_g, axis=0)
        sgn = np.tile(np.r_[np.ones(nk), -np.ones(nk)], N)
        kk = np.tile(np.r_[np.arange(nk), np.arange(nk)], N)
        zz[np.arange(N * m_g), kk] += sgn * dz
        og = forward(Pg, calg, rg, h_from_knots(zz, T), Qg)
        Vg, _ = lab_value(og, rg, hz)
        Vg = Vg.reshape(N, 2, nk)
        grad = (Vg[:, 0] - Vg[:, 1]) / (2 * dz)
        scale = np.abs(V)[:, None] + 1e-12
        d = grad / scale                                   # relative gradient
        d = d / (np.abs(d).max(axis=1, keepdims=True) + 1e-12)   # unit max-norm direction
        zc = np.repeat(z, m_s, axis=0) + np.tile(steps, N)[:, None] * np.repeat(d, m_s, axis=0)
        zc = np.clip(zc, -8, 8)
        oc = forward(Ps, cals, rs, h_from_knots(zc, T), Qs)
        Vc, _ = lab_value(oc, rs, hz)
        Vc = np.where(np.isfinite(Vc), Vc, -np.inf).reshape(N, m_s)
        best = Vc.argmax(axis=1)
        z = zc.reshape(N, m_s, nk)[np.arange(N), best]
        gain = (Vc[np.arange(N), best] - V) / (np.abs(V) + 1e-12)
        V = Vc[np.arange(N), best]
        if verbose:
            print(f"      lab it {it:2d}: rel. PV gain {gain.max():.2e}")
        if gain.max() < 1e-5:
            break
    # relative marginal value of each knot at the optimum (for the KKT check)
    return z, grad / (np.abs(V)[:, None] + 1e-12)


def solve_equilibrium(P, n_outer=20, tol_h=0.01, verbose=True, r_init=None):
    """Full perfect-foresight equilibrium:
       inner : Newton on interest rates (Euler equation) given lab policy
       outer : lab policy maximises PV of lab profits given prices."""
    T, N = P["T"], P["N"]
    cal = calibrate(P)
    z = np.full((N, len(KNOTS)), np.log((P["h0"][0] - H_LO) / (H_HI - P["h0"][0])))
    h = h_from_knots(z, T)
    r = None if r_init is None else r_init.copy()
    Qpath = None
    hist = []
    for k in range(n_outer):
        r, Qpath, o = solve_rates(P, cal, h, r, Qpath)
        bad = o["euler_gap"] > 1e-4
        if bad.any():                                  # restart Newton from the initial guess
            r2, Q2, o2 = solve_rates(P, cal, h, None if r_init is None else r_init.copy(), None)
            if o2["euler_gap"].max() < o["euler_gap"].max():
                r, Qpath, o = r2, Q2, o2
        z_br, gk = lab_optimize(P, cal, r, Qpath, z)   # lab best response
        omega = max(0.5 * 0.85 ** k, 0.15)             # damping of the outer fixed point
        z = z + omega * (z_br - z)
        h_new = h_from_knots(z, T)
        dh = np.abs(h_new - h)[:, :51].max(axis=1)
        h = h_new
        hist.append(dh.max())
        if verbose:
            print(f"  outer {k:2d}: max change in lab R&D share = {dh.max():.4f}, "
                  f"Euler gap = {o['euler_gap'].max():.1e}")
        if dh.max() < tol_h:
            break
    r, Qpath, o = solve_rates(P, cal, h, r, Qpath)
    o["years"] = 2025 + np.arange(T + 1)
    o["cal"] = cal
    o["h_knots"] = z
    o["lab_grad_knots"] = gk
    o["outer_dh"] = dh
    o["hist"] = hist
    return o


def draw_parameters(N=100, seed=7):
    rng = np.random.default_rng(seed)
    d = dict(
        eps=rng.uniform(0.3, 0.95, N),
        r_S=rng.uniform(0.5, 1.5, N),
        beta_max=rng.uniform(0.6, 0.995, N),
        tau=rng.uniform(3.0, 15.0, N),
        g_P=rng.uniform(0.05, 0.30, N),
        kappa_mid=rng.uniform(8.0, 18.0, N),
        theta=rng.uniform(1.0, 2.5, N),
        mu=rng.uniform(1.5, 3.0, N),
    )
    d["betaR_max"] = d["beta_max"]
    return d


def initial_rates(P, d):
    """Initial interest-rate guess r = rho + theta * g, with g from a v1.0
    simulation with the same technology parameters (cheap and robust)."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "v1.0", "src"))
    import ai_growth_model as v1
    N, T = P["N"], P["T"]
    kw = {("r" if k == "r_S" else k): v for k, v in d.items() if k not in ("theta", "mu")}
    o1 = v1.simulate(N=N, T1=2025 + T + 1, **kw)
    idx = [np.argmin(abs(o1["years"] - (2025 + t))) for t in range(T + 2)]
    lnY = np.log(o1["Y"][idx].T)
    g = np.diff(lnY, axis=1, prepend=lnY[:, :1])
    r = P["rho_disc"][:, None] + P["theta"][:, None] * np.clip(g, 0.0, 2.0)
    return np.clip(r, 0.02, 5.0)


def _solve_chunk(args):
    d, = args
    N = len(d["eps"])
    P = make_params(N, **d)
    with np.errstate(all="ignore"):
        o = solve_equilibrium(P, verbose=False, r_init=initial_rates(P, d))
    keep = ["Y", "g", "r", "save_rate", "AI_capex_share", "h", "beta", "lab_share", "w",
            "c_pc", "euler_gap", "outer_dh", "power_bind"]
    return {k: o[k] for k in keep}


def monte_carlo(N=100, seed=7, chunk=10, processes=2):
    """Solve N parameter draws in chunks (in parallel processes)."""
    from multiprocessing import Pool
    d = draw_parameters(N, seed)
    chunks = [({k: v[i:i + chunk] for k, v in d.items()},) for i in range(0, N, chunk)]
    with Pool(processes) as pool:
        res = pool.map(_solve_chunk, chunks)
    o = {k: np.concatenate([r_[k] for r_ in res]) for k in res[0]}
    o["years"] = 2025 + np.arange(o["Y"].shape[1])
    return d, o
