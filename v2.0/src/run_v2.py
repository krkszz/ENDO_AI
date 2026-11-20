"""
ENDO_AI v2.0 -- solve the three scenarios, compare with v1.0, analyse the
Monte Carlo (run run_mc_v2.py first), and write all results to ../output.
"""
import csv
import os
import pickle
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "v1.0", "src"))
from endo_ai_v2 import make_params, solve_equilibrium, SCENARIOS  # noqa: E402
import ai_growth_model as v1                                          # noqa: E402

OUT = os.path.join(HERE, "..", "output")
os.makedirs(OUT, exist_ok=True)
P_ = lambda f: os.path.join(OUT, f)
np.seterr(all="ignore")

COL = {"Bottleneck": "#2a78d6", "Transformative": "#eb6834", "Explosive": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})
END = 2075
names = list(SCENARIOS)

# ---------------------------------------------------------------- scenarios
cache = P_("scenario_cache.pkl")
if os.path.exists(cache) and "--resolve" not in sys.argv:
    o = pickle.load(open(cache, "rb"))
else:
    t0 = time.time()
    P = make_params(3, **{k: np.array([SCENARIOS[s][k] for s in names]) for k in SCENARIOS[names[0]]})
    o = solve_equilibrium(P)
    o.pop("cal", None)
    pickle.dump(o, open(cache, "wb"))
    print(f"scenarios solved in {time.time() - t0:.0f} s")
yrs = o["years"]
m = yrs <= END
o1 = {s: v1.simulate(**{("r" if k == "r_S" else k): v for k, v in SCENARIOS[s].items()}) for s in names}


def v1_annual(s, key):
    oo = o1[s]
    idx = [np.argmin(abs(oo["years"] - y)) for y in yrs[m]]
    return oo[key][idx, 0]


def label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points",
                color=INK2, fontsize=8.5, va="center")


# Figure 1: equilibrium paths
fig, ax = plt.subplots(3, 2, figsize=(11, 10.5))
panels = [
    ("g", "GDP growth (% per year, capped at 100%)", lambda v: np.minimum(100 * v, 100), "linear"),
    ("r", "Real interest rate (% per year, log scale)", lambda v: 100 * v, "log"),
    ("save_rate", "Gross saving rate (% of GDP)", lambda v: 100 * v, "linear"),
    ("AI_capex_share", "AI investment: chips + power (% of GDP)", lambda v: 100 * v, "linear"),
    ("h", "Labs' share of compute used for R&D (%)", lambda v: 100 * v, "linear"),
    ("beta", "Share of goods tasks automated (%)", lambda v: 100 * v, "linear"),
]
for a, (key, title, f, scale) in zip(ax.flat, panels):
    for i, s in enumerate(names):
        v = f(o[key][i])
        a.plot(yrs[m][1:] if key in ("g", "r") else yrs[m], (v[m][1:] if key in ("g", "r") else v[m]),
               color=COL[s], lw=2, label=s)
    a.set_title(title, loc="left")
    a.set_yscale(scale)
    a.set_xlim(2025, END)
ax[0, 0].legend(frameon=False, loc="upper left")
fig.suptitle("ENDO_AI v2.0 — perfect-foresight equilibrium, three scenarios", x=0.01, ha="left",
             fontsize=13, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(P_("fig1_equilibrium_paths.png"), dpi=160)
plt.close(fig)

# Figure 2: v1 vs v2
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
for j, s in enumerate(names):
    a = ax[0, j]
    a.plot(yrs[m][1:], np.minimum(100 * o["g"][j][m][1:], 100), color=COL[s], lw=2, label="v2.0 (forward-looking)")
    a.plot(yrs[m][1:], np.minimum(100 * v1_annual(s, "g")[1:], 100), color=INK2, lw=1.5, ls="--", label="v1.0 (rules)")
    a.set_title(f"{s}: GDP growth (%)", loc="left")
    a = ax[1, j]
    a.plot(yrs[m], 100 * o["AI_capex_share"][j][m], color=COL[s], lw=2, label="v2.0")
    a.plot(yrs[m], 100 * v1_annual(s, "AI_capex_share"), color=INK2, lw=1.5, ls="--", label="v1.0")
    a.set_title(f"{s}: AI investment (% of GDP)", loc="left")
    a.set_xlim(2025, END)
ax[0, 0].legend(frameon=False, loc="upper left")
fig.suptitle("v2.0 versus v1.0", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(P_("fig2_v1_vs_v2.png"), dpi=160)
plt.close(fig)

# Figure 3: households and labour
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for i, s in enumerate(names):
    w = o["w"][i] / o["w"][i][0]
    c = o["c_pc"][i] / o["c_pc"][i][0]
    ax[0].plot(yrs[m], w[m], color=COL[s], lw=2, label=s)
    ax[1].plot(yrs[m], c[m], color=COL[s], lw=2, label=s)
    ax[2].plot(yrs[m], 100 * o["lab_share"][i][m], color=COL[s], lw=2, label=s)
for a, t in zip(ax, ["Real wage (2025 = 1, log scale)", "Consumption per head (2025 = 1, log scale)",
                     "Labour share of income (%)"]):
    a.set_title(t, loc="left"); a.set_xlim(2025, END)
ax[0].set_yscale("log"); ax[1].set_yscale("log")
ax[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(P_("fig3_wages_consumption.png"), dpi=160)
plt.close(fig)

# scenario CSV
keys = ["Y", "g", "r", "C", "c_pc", "save_rate", "AI_capex_share", "I_K", "I_chip", "I_P",
        "h", "beta", "betaR", "lnS", "w", "lab_share", "lab_profit", "R_H", "UC", "price_AI",
        "H", "K", "power_bind"]
with open(P_("scenario_paths.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["scenario", "year"] + keys)
    for i, s in enumerate(names):
        for t in np.where(m)[0]:
            wr.writerow([s, int(yrs[t])] + [f"{o[k][i, t]:.6g}" for k in keys])

# ---------------------------------------------------------------- summary
L = []
L.append("ENDO_AI v2.0 - perfect-foresight equilibrium")
L.append("")
L.append("Convergence (scenarios): max Euler gap %s ; max change of lab policy in last outer step %s"
         % (np.array2string(o["euler_gap"], precision=1), np.array2string(o["outer_dh"], precision=4)))
L.append("")
L.append("Scenario results (v2.0 | v1.0 in brackets):")


def at(arr, y):
    return arr[np.argmin(abs(yrs - y))]


for i, s in enumerate(names):
    g = o["g"][i]
    avg = np.log(at(o["Y"][i], 2050) / o["Y"][i][0]) / 25
    v1g = o1[s]["Y"][:, 0]
    v1avg = np.log(v1g[np.argmin(abs(o1[s]["years"] - 2050))] / v1g[0]) / 25
    L.append(f"{s}:")
    L.append(f"  GDP growth 2030 / 2040 / 2050: {100*at(g,2030):.1f}% / {100*at(g,2040):.1f}% / {100*at(g,2050):.1f}%"
             f"   [v1: {100*v1_annual(s,'g')[5]:.1f}% / {100*v1_annual(s,'g')[15]:.1f}% / {100*v1_annual(s,'g')[25]:.1f}%]")
    L.append(f"  average growth 2025-2050: {100*avg:.1f}%  [v1: {100*v1avg:.1f}%]")
    L.append(f"  interest rate 2030 / 2040 / 2050: {100*at(o['r'][i],2030):.1f}% / {100*at(o['r'][i],2040):.1f}% / {100*at(o['r'][i],2050):.1f}%")
    L.append(f"  saving rate 2025 / 2035 / 2050: {100*o['save_rate'][i][0]:.1f}% / {100*at(o['save_rate'][i],2035):.1f}% / {100*at(o['save_rate'][i],2050):.1f}%")
    L.append(f"  AI investment, peak share of GDP to 2075: {100*o['AI_capex_share'][i][m].max():.2f}% in {int(yrs[m][o['AI_capex_share'][i][m].argmax()])}")
    L.append(f"  lab R&D share 2025 / 2035 / 2050: {100*o['h'][i][0]:.0f}% / {100*at(o['h'][i],2035):.0f}% / {100*at(o['h'][i],2050):.0f}%")
    L.append(f"  tasks automated 2050: {100*at(o['beta'][i],2050):.0f}%   wage index 2050: {at(o['w'][i],2050)/o['w'][i][0]:.3g}"
             f"   labour share 2050: {100*at(o['lab_share'][i],2050):.0f}%")

# ---------------------------------------------------------------- Monte Carlo
mc_file = P_("mc_cache.pkl")
if os.path.exists(mc_file):
    mc = pickle.load(open(mc_file, "rb"))
    d, om = mc["draws"], mc["o"]
    ok = (om["euler_gap"] < 1e-4) & (om["outer_dh"] < 0.05) & np.isfinite(om["Y"][:, 50])
    y = om["years"]
    i50 = np.argmin(abs(y - 2050)); i75 = np.argmin(abs(y - 2075)); i30 = np.argmin(abs(y - 2030))
    avg50 = np.log(om["Y"][:, i50] / om["Y"][:, 0]) / 25
    avg75 = np.log(om["Y"][:, i75] / om["Y"][:, i50]) / 25
    capex30 = om["AI_capex_share"][:, :i30 + 1].mean(axis=1)
    s0 = om["save_rate"][:, 0]
    pnames = {"kappa_mid": "Capability difficulty κ_mid", "eps": "Task elasticity ε",
              "r_S": "Software returns r_S", "tau": "Adoption lag τ", "g_P": "Power build-out g_P",
              "beta_max": "Max automation β_max", "theta": "Inverse IES θ", "mu": "Lab markup μ"}
    ks = list(pnames)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    bins = np.linspace(0, 40, 41)
    ax[0].hist(np.clip(100 * avg50[ok], 0, 40), bins=bins, color="#2a78d6", edgecolor="#fcfcfb", lw=1.5, label="2025–2050")
    ax[0].hist(np.clip(100 * avg75[ok], 0, 40), bins=bins, color="#eb6834", alpha=0.55, edgecolor="#fcfcfb", lw=1.5, label="2050–2075")
    txt = []
    for arr, per in [(avg50[ok], "2025–50"), (avg75[ok], "2050–75")]:
        p10, p50, p90 = np.percentile(100 * arr, [10, 50, 90])
        txt.append(f"{per}: median {p50:.1f}%, p10–p90 {p10:.1f}–{p90:.1f}%")
    ax[0].text(0.98, 0.70, "\n".join(txt), transform=ax[0].transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    ax[0].set_xlabel("average annual GDP growth (%; >40% binned at 40)")
    ax[0].set_ylabel("number of draws")
    ax[0].set_title(f"Distribution over {ok.sum()} converged draws", loc="left")
    ax[0].legend(frameon=False)

    ax[1].scatter(100 * capex30[ok], 100 * avg50[ok], s=18, color="#2a78d6", edgecolor="#fcfcfb", lw=0.6)
    ax[1].set_xlabel("average AI investment 2025–2030 (% of GDP)")
    ax[1].set_ylabel("average GDP growth 2025–2050 (%)")
    ax[1].set_title("Investment today anticipates growth later", loc="left")
    rho_c = spearmanr(capex30[ok], avg50[ok]).correlation
    ax[1].text(0.03, 0.95, f"Spearman ρ = {rho_c:.2f}", transform=ax[1].transAxes, va="top", fontsize=9, color=INK2)

    c50 = [spearmanr(d[k][ok], avg50[ok]).correlation for k in ks]
    c75 = [spearmanr(d[k][ok], avg75[ok]).correlation for k in ks]
    order = np.argsort(np.abs(c50)); yy = np.arange(len(ks))
    ax[2].barh(yy + 0.2, np.array(c50)[order], height=0.38, color="#2a78d6", label="growth 2025–50")
    ax[2].barh(yy - 0.2, np.array(c75)[order], height=0.38, color="#eb6834", label="growth 2050–75")
    ax[2].set_yticks(yy, [pnames[ks[k]] for k in order])
    ax[2].axvline(0, color=INK2, lw=0.8)
    ax[2].set_xlabel("Spearman rank correlation with growth")
    ax[2].set_title("What drives the outcome?", loc="left")
    ax[2].legend(frameon=False, loc="lower right")
    ax[2].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(P_("fig4_monte_carlo.png"), dpi=160)
    plt.close(fig)

    with open(P_("monte_carlo_draws.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        pk = list(d)
        wr.writerow(pk + ["converged", "avg_growth_2025_50", "avg_growth_2050_75", "ai_capex_2025_30",
                          "saving_rate_2025", "r_2030", "labour_share_2050", "euler_gap", "outer_dh"])
        for j in range(len(avg50)):
            wr.writerow([f"{d[k][j]:.6g}" for k in pk] + [int(ok[j]), f"{avg50[j]:.6g}", f"{avg75[j]:.6g}",
                         f"{capex30[j]:.6g}", f"{s0[j]:.6g}", f"{om['r'][j, i30]:.6g}",
                         f"{om['lab_share'][j, i50]:.6g}", f"{om['euler_gap'][j]:.3g}", f"{om['outer_dh'][j]:.3g}"])

    L.append("")
    L.append(f"Monte Carlo: {len(avg50)} draws, {ok.sum()} converged (Euler gap < 1e-4, lab policy change < 0.05)")
    L.append("Average growth 2025-2050, percentiles 5/25/50/75/95 (%%): %s"
             % np.round(100 * np.percentile(avg50[ok], [5, 25, 50, 75, 95]), 1).tolist())
    L.append("Average growth 2050-2075, percentiles 5/25/50/75/95 (%%): %s"
             % np.round(100 * np.percentile(avg75[ok], [5, 25, 50, 75, 95]), 1).tolist())
    L.append("Share of draws with average growth > 10%% in 2025-2050: %.3f" % (avg50[ok] > 0.10).mean())
    L.append("Average AI investment 2025-2030, percentiles 5/50/95 (%% of GDP): %s"
             % np.round(100 * np.percentile(capex30[ok], [5, 50, 95]), 2).tolist())
    L.append("Spearman correlation of early AI investment with later growth: %.2f" % rho_c)
    L.append("")
    L.append("%-12s %10s %10s" % ("parameter", "2025-50", "2050-75"))
    for k, a5, a7 in zip(ks, c50, c75):
        L.append("%-12s %+10.2f %+10.2f" % (k, a5, a7))

text = "\n".join(L)
print(text)
open(P_("summary.txt"), "w").write(text + "\n")
