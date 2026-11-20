"""
Endogenous growth model with AI -- Version 1 (rule-based closure)
=================================================================

Task-based production with endogenous automation, two research sectors
(general TFP A and AI software S) that AI can partly automate,
compute hardware H with a price-performance learning curve, an energy
cap on usable compute, adoption lags, and investment allocated between
physical capital and compute by comparing returns.

Every state is a numpy array of length N, so one call simulates N parameter
draws at once (used for the Monte Carlo).

Units: world GDP in 2025 = 100 (roughly $ trillion), population L = 1.
Time step dt = 0.1 years, horizon 2025-2075.

Closure (Version 1):
  * gross saving rate s fixed
  * research labour shares and compute-use shares fixed
  * investment split between K and H by partial adjustment toward the
    share that equalises net returns (softmax over returns)
  * automation is adopted only where AI is cheaper than human labour
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Default parameters (all can be overridden; arrays allowed for Monte Carlo)
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    # horizon
    T0=2025.0, T1=2075.0, dt=0.1,
    # goods production
    alpha=0.35,          # capital share in Cobb-Douglas outer nest
    eps=0.6,             # elasticity of substitution between tasks (key!)
    delta_K=0.05,
    s=0.25,              # gross saving rate
    gK_max=0.30,         # max net growth of physical capital (construction limit)
    n=0.004,             # population growth
    # automation (capability -> share of tasks AI can do)
    beta0=0.003,         # share of goods tasks automated in 2025
    beta_max=0.95,       # long-run ceiling on automatable goods tasks
    betaR0=0.003,        # share of research tasks automated in 2025
    betaR_max=0.95,      # ceiling on automatable research tasks
    kappa_mid=12.0,      # log-units of cumulative effective training compute
                         # (above 2025 level) until half of beta_max is reached
    kappaR_extra=3.0,    # research is somewhat harder to automate
    tau=8.0,             # adoption lag for goods tasks (years)
    tau_R=3.0,           # adoption lag for research tasks (labs adopt fast)
    gate_slope=2.0,      # how sharply adoption responds to AI cost vs wage
    # compute / hardware
    V_H0=1.0,            # 2025 value of AI hardware stock (~$1T)
    I_H_share0=0.01,     # 2025 share of market investment to inference compute
    delta_H=0.20,        # hardware depreciation (obsolescence)
    g_Q=0.25,            # exogenous log-growth of FLOP/s per $ (~2.8y doubling)
    wright_b=0.10,       # learning-by-doing elasticity on cumulative chip spend
    zeta=3.0,            # sensitivity of investment split to return gap
    tau_I=2.0,           # adjustment lag of investment split (years)
    xH_max=0.6,          # max share of market investment to inference compute
    # frontier-lab spending (training, research, experiments), share of GDP
    m0=0.004,            # ~$400bn/yr in 2025
    m_max=0.03,          # long-run lab compute spending share of GDP
    m_mid=2032.0, m_w=3.0,
    g_LS=0.08, LS_mult_max=5.0,   # growth and cap of human AI-research labour
    # compute uses (fixed shares)
    u_T=0.30, u_Y=0.40, u_RA=0.10, u_RS=0.15, u_exp=0.05,
    c_R=30.0,            # AI researcher needs c_R x the compute of an AI worker
    # energy
    E0_mult=2.0,         # 2025 power capacity can run 2.0x the 2025 chip stock
    g_eta=0.12,          # log-growth of FLOP per joule
    g_P=0.15,            # MAX log-growth of power capacity for AI (build-out limit)
    power_cost=0.5,      # 2025 cost of power+datacentre per FLOP/s, rel. to chips
    delta_P=0.04,        # depreciation of power/datacentre capacity
    # general TFP research: A_dot/A = th_A a^(phi_A-1) (R_A/R_A0)^lam_A
    th_A=0.012, lam_A=0.5, phi_A=0.5,
    LA_share=0.015,
    # AI software research
    th_S=0.9,            # ~2.5x/yr algorithmic efficiency in 2025
    lam_S=0.6,
    r=1.0,               # returns to software research r = lam_S/(1-phi_S) (key!)
    gam=0.3,             # elasticity w.r.t. experiment compute (raw hardware)
    lnS_max=np.log(1e6), # physical limit on software efficiency (x1e6)
    LS_share=0.0005,
)


def _ces(xa, xl, b, eps):
    """CES aggregate of AI input xa and labour xl with automated share b.
    Returns (aggregate Z, inner sum B, rho)."""
    rho = (eps - 1.0) / eps
    B = b ** (1.0 / eps) * xa ** rho + (1.0 - b) ** (1.0 / eps) * xl ** rho
    return B ** (1.0 / rho), B, rho


def _logistic(k, mid, width, top):
    return top / (1.0 + np.exp(-(k - mid) / width))


def simulate(N: int = 1, **kw):
    p = {**DEFAULTS, **kw}
    P = {k: (np.broadcast_to(np.asarray(v, float), (N,)).copy()
             if k not in ("T0", "T1", "dt") else v) for k, v in p.items()}
    dt = P["dt"]
    years = np.arange(P["T0"], P["T1"] + 1e-9, dt)
    nT = len(years)

    alpha, eps = P["alpha"], P["eps"]
    phi_S = 1.0 - P["lam_S"] / P["r"]

    # capability curves: calibrate width so beta(kappa=0) = beta0
    w_Y = P["kappa_mid"] / np.log(P["beta_max"] / P["beta0"] - 1.0)
    midR = P["kappa_mid"] + P["kappaR_extra"]
    w_R = midR / np.log(P["betaR_max"] / P["betaR0"] - 1.0)

    # ---------------- initial conditions ----------------
    Y0 = 100.0
    L = np.ones(N)
    K = 3.0 * Y0 * np.ones(N)
    LY = L * (1 - P["LA_share"] - P["LS_share"])
    bt = P["beta0"].copy()          # deployed goods automation
    btR = P["betaR0"].copy()        # deployed research automation
    X0 = P["beta0"] * LY            # AI worker-equivalents ~ same intensity as humans
    H0 = np.ones(N)                 # FLOP/s units, normalised
    c = H0 * P["u_Y"] / X0          # compute per AI worker-equivalent (S0 = 1)
    Q = H0 / P["V_H0"]              # FLOP/s per $
    H = H0.copy()
    lnS = np.zeros(N)
    a = np.ones(N)
    eta = np.ones(N)                # FLOP/s per unit of power, normalised
    Pw = P["E0_mult"] * H0          # power capacity (in 2025 FLOP/s-equivalents)
    p_P = P["power_cost"] / Q       # $ per unit of power capacity
    cumI_H = P["V_H0"] * 3.0        # rough cumulative chip spend to date
    cumI_H = cumI_H * np.ones(N)
    xH = P["I_H_share0"].copy()
    cumCT = np.ones(N)              # cumulative effective training compute, normalised
    CT_norm = None

    # calibrate A0 so Y0 = 100
    Z0, _, _ = _ces(X0, LY, bt, eps)
    A0 = Y0 / (K ** alpha * Z0 ** (1 - alpha))

    # research input normalisations
    LA0 = P["LA_share"] * L
    LS0 = P["LS_share"] * L
    S = np.exp(lnS)
    XRA0 = S * H * P["u_RA"] / (c * P["c_R"])
    XRS0 = S * H * P["u_RS"] / (c * P["c_R"])
    RA0, _, _ = _ces(XRA0, LA0, btR, eps)
    RS0, _, _ = _ces(XRS0, LS0, btR, eps)
    Hexp0 = H * P["u_exp"]

    out = {k: np.empty((nT, N)) for k in
           ["Y", "g", "w", "lab_share", "beta", "betaR", "xH", "gS", "gA",
            "H", "E_bind", "AI_capex_share", "lnS", "K", "C_pc"]}
    Y_prev = None

    for t in range(nT):
        S = np.exp(lnS)
        E = eta * Pw
        H_use = np.minimum(H, E)
        pi = 1.0 / Q + p_P / eta        # $ per unit of new *usable* compute

        X = S * H_use * P["u_Y"] / c
        Z, B, rho = _ces(X, LY, bt, eps)
        A = A0 * a
        Y = A * K ** alpha * Z ** (1 - alpha)

        # factor prices
        w = (1 - alpha) * Y * (1 - bt) ** (1 / eps) * LY ** (rho - 1) / B
        MPX = (1 - alpha) * Y * bt ** (1 / eps) * X ** (rho - 1) / B
        MPK = alpha * Y / K
        ret_K = MPK - P["delta_K"]
        # return per $ of hardware: value of the X it yields in goods use
        # (all compute uses assumed to rent at that rate); pi = chips + power
        ret_H = MPX * (S / c) / pi - P["delta_H"]
        # user cost of one AI worker-equivalent vs wage
        p_X = (np.maximum(ret_K, 0.01) + P["delta_H"]) * pi / (S / c)

        # ---- record ----
        out["Y"][t] = Y
        out["g"][t] = np.nan if Y_prev is None else np.log(Y / Y_prev) / dt
        out["w"][t] = w
        out["lab_share"][t] = w * L / Y
        out["beta"][t] = bt
        out["betaR"][t] = btR
        out["xH"][t] = xH
        out["H"][t] = H
        out["lnS"][t] = lnS
        out["K"][t] = K
        out["C_pc"][t] = (1 - P["s"]) * Y / L
        Y_prev = Y

        # ---- investment allocation ----
        # (i) frontier labs buy compute for training/research as a GDP share
        #     (expectations-driven rule; Version 2 would derive it)
        yr = years[t]
        m = P["m0"] + (P["m_max"] - P["m0"]) / (1 + np.exp(-(yr - P["m_mid"]) / P["m_w"]))
        I_lab = m * Y
        # (ii) the rest of saving is split between K and inference hardware
        #      by comparing net returns (partial adjustment)
        I_mkt = np.maximum(P["s"] * Y - I_lab, 0.0)
        rk = np.maximum(ret_K, 1e-4)
        rh = np.maximum(ret_H, 1e-4)
        target = 1.0 / (1.0 + (rk / rh) ** P["zeta"])
        target = np.minimum(target, P["xH_max"])
        xH = xH + (target - xH) * dt / P["tau_I"]
        I_C = I_lab + xH * I_mkt           # desired spending on compute (chips + power)
        # chips and power are bought together; power build-out is rate-limited
        dH = I_C / pi * dt                 # new usable compute if unconstrained
        eta_next = eta * np.exp(P["g_eta"] * dt)
        need_P = (H * (1 - P["delta_H"] * dt) + dH) / eta_next
        P_max = Pw * (1 - P["delta_P"] * dt) * np.exp(P["g_P"] * dt)
        bind = need_P > P_max
        # if power binds, scale chip purchases down to what can be powered
        dH = np.where(bind, np.maximum(P_max * eta_next - H * (1 - P["delta_H"] * dt), 0.0), dH)
        Pw_new = np.maximum(np.minimum(need_P, P_max), Pw * (1 - P["delta_P"] * dt))
        I_P = np.maximum(Pw_new - Pw * (1 - P["delta_P"] * dt), 0.0) * p_P / dt
        I_chip = dH / Q / dt
        I_H = I_chip + I_P
        I_K = P["s"] * Y - I_H
        out["AI_capex_share"][t] = I_H / Y
        out["E_bind"][t] = bind.astype(float)

        # ---- capability & automation ----
        CT = S * H_use * P["u_T"]
        if CT_norm is None:
            CT_norm = CT * 3.0          # 2025 cumulative ~ 3 years of current
            cumCT = np.ones(N)
        cumCT = cumCT + CT / CT_norm * dt
        kappa = np.log(cumCT)
        beta_cap = _logistic(kappa, P["kappa_mid"], w_Y, P["beta_max"])
        betaR_cap = _logistic(kappa, midR, w_R, P["betaR_max"])
        gate = 1.0 / (1.0 + np.exp(-P["gate_slope"] * np.log(w / p_X)))
        tgt_Y = np.maximum(beta_cap * gate, P["beta0"])
        bt = bt + (tgt_Y - bt) * dt / P["tau"]
        bt = np.clip(bt, 1e-6, 1 - 1e-6)
        btR = btR + (betaR_cap - btR) * dt / P["tau_R"]
        btR = np.clip(btR, 1e-6, 1 - 1e-6)

        # ---- research ----
        LA = P["LA_share"] * L
        LS = P["LS_share"] * L * np.minimum(np.exp(P["g_LS"] * (years[t] - P["T0"])), P["LS_mult_max"])
        XRA = S * H_use * P["u_RA"] / (c * P["c_R"])
        XRS = S * H_use * P["u_RS"] / (c * P["c_R"])
        RA, _, _ = _ces(XRA, LA, btR, eps)
        RS, _, _ = _ces(XRS, LS, btR, eps)
        gA = P["th_A"] * a ** (P["phi_A"] - 1) * (RA / RA0) ** P["lam_A"]
        ceiling = np.clip(1.0 - lnS / P["lnS_max"], 0.0, 1.0)
        gS = (P["th_S"] * S ** (phi_S - 1) * (RS / RS0) ** P["lam_S"]
              * (H_use * P["u_exp"] / Hexp0) ** P["gam"] * ceiling)
        gS = np.minimum(gS, 5.0)        # numerical guard (148x per year)
        out["gS"][t] = gS
        out["gA"][t] = gA

        # ---- state updates ----
        a = a * np.exp(gA * dt)
        lnS = np.minimum(lnS + gS * dt, P["lnS_max"])
        cumI_H = cumI_H + I_chip * dt
        Q = Q * np.exp(P["g_Q"] * dt) * (1 + I_chip * dt / cumI_H) ** P["wright_b"]
        H = H * (1 - P["delta_H"] * dt) + dH
        Pw = Pw_new
        eta = eta_next
        K = np.minimum(K + (I_K - P["delta_K"] * K) * dt, K * np.exp(P["gK_max"] * dt))
        L = L * np.exp(P["n"] * dt)
        LY = L * (1 - P["LA_share"] - P["LS_share"])

    out["years"] = years
    return out


# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------
SCENARIOS = {
    "Bottleneck": dict(eps=0.4, r=0.7, beta_max=0.70, betaR_max=0.60,
                       kappa_mid=16.0, tau=12.0, g_P=0.08, m_max=0.012),
    "Transformative": dict(eps=0.6, r=1.0, beta_max=0.95, betaR_max=0.95,
                           kappa_mid=12.0, tau=7.0, g_P=0.15, m_max=0.03),
    "Explosive": dict(eps=0.8, r=1.4, beta_max=0.995, betaR_max=0.995,
                      kappa_mid=9.0, tau=4.0, g_P=0.30, m_max=0.05),
}


def annual(series, years):
    """Sample a dt-grid series at whole years."""
    idx = np.where(np.isclose(years % 1.0, 0.0) | np.isclose(years % 1.0, 1.0))[0]
    return years[idx], series[idx]


def monte_carlo(N=4000, seed=1):
    rng = np.random.default_rng(seed)
    draws = dict(
        eps=rng.uniform(0.3, 0.95, N),
        r=rng.uniform(0.5, 1.5, N),
        beta_max=rng.uniform(0.6, 0.995, N),
        tau=rng.uniform(3.0, 15.0, N),
        g_P=rng.uniform(0.05, 0.30, N),
        kappa_mid=rng.uniform(8.0, 18.0, N),
        m_max=rng.uniform(0.01, 0.05, N),
    )
    draws["betaR_max"] = draws["beta_max"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        res = simulate(N=N, **draws)
    return draws, res


if __name__ == "__main__":
    for name, kw in SCENARIOS.items():
        o = simulate(**kw)
        yrs = o["years"]
        def at(y, k): return o[k][np.argmin(abs(yrs - y)), 0]
        g = o["g"][:, 0]
        print(f"{name:15s} g2030={at(2030,'g'):.3f} g2040={at(2040,'g'):.3f} "
              f"g2050={at(2050,'g'):.3f} g2070={at(2070,'g'):.3f} "
              f"peak={np.nanmax(g):.3f} Y2050={at(2050,'Y'):.0f} "
              f"labsh2025={at(2025,'lab_share'):.2f} labsh2050={at(2050,'lab_share'):.2f} "
              f"beta2050={at(2050,'beta'):.2f} capex2030={at(2030,'AI_capex_share'):.3f} "
              f"w2050/w2025={at(2050,'w')/at(2025,'w'):.2f}")
