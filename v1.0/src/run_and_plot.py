"""Run the three scenarios and the Monte Carlo; write figures and a CSV."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from ai_growth_model import simulate, monte_carlo, SCENARIOS

# all results go to ../output relative to this file
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUT, exist_ok=True)
P = lambda f: os.path.join(OUT, f)

COL = {"Bottleneck": "#2a78d6", "Transformative": "#eb6834", "Explosive": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})

# ---------------------------------------------------------------- scenarios
runs = {k: simulate(**v) for k, v in SCENARIOS.items()}
yrs = runs["Bottleneck"]["years"]
m = yrs >= 2025.5

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
panels = [
    ("g", "GDP growth rate (% per year, capped at 100%)", lambda v: np.minimum(100 * v, 100), "linear"),
    ("beta", "Share of goods tasks automated (%)", lambda v: 100 * v, "linear"),
    ("w", "Real wage, index 2025 = 1 (log scale)", lambda v: v / v[0], "log"),
    ("lab_share", "Labour share of income (%)", lambda v: 100 * v, "linear"),
]
for a, (key, title, f, scale) in zip(ax.flat, panels):
    for name, o in runs.items():
        v = f(o[key][:, 0])
        a.plot(yrs[m], v[m], color=COL[name], lw=2, label=name)
        lab = name
        if key == "lab_share" and name != "Explosive":
            if name == "Bottleneck":
                continue          # the two lines coincide; label once
            lab = "Bottleneck &\nTransformative"
        a.annotate(lab, (yrs[-1], v[-1]), xytext=(4, 0), textcoords="offset points",
                   color=INK2, fontsize=8.5, va="center")
    a.set_title(title, loc="left")
    a.set_yscale(scale)
    a.set_xlim(2025, 2083)
ax[0, 0].legend(frameon=False, loc="upper left")
fig.suptitle("Three AI scenarios, 2025–2075", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(P("fig1_scenarios.png"), dpi=160)

# CSV of annual scenario paths
with open(P("scenario_paths.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    keys = ["Y", "g", "beta", "betaR", "w", "lab_share", "AI_capex_share", "gS", "gA", "E_bind"]
    wr.writerow(["scenario", "year"] + keys)
    for name, o in runs.items():
        for i in np.where(np.isclose(yrs % 1, 0) | np.isclose(yrs % 1, 1))[0]:
            wr.writerow([name, int(round(yrs[i]))] + [f"{o[k][i, 0]:.6g}" for k in keys])

# ---------------------------------------------------------------- Monte Carlo
draws, mc = monte_carlo(N=6000)
y = mc["years"]
i50, i75 = np.argmin(abs(y - 2050)), np.argmin(abs(y - 2075))
avg50 = np.log(mc["Y"][i50] / mc["Y"][0]) / 25
avg75 = np.log(mc["Y"][i75] / mc["Y"][i50]) / 25
ls50 = mc["lab_share"][i50]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

# (a) heatmap eps x r -> median average growth 2025-2050
eb = np.linspace(0.3, 0.95, 14)
rb = np.linspace(0.5, 1.5, 11)
H = np.full((len(rb) - 1, len(eb) - 1), np.nan)
for i in range(len(rb) - 1):
    for j in range(len(eb) - 1):
        sel = ((draws["r"] >= rb[i]) & (draws["r"] < rb[i + 1]) &
               (draws["eps"] >= eb[j]) & (draws["eps"] < eb[j + 1]))
        if sel.sum() >= 5:
            H[i, j] = 100 * np.median(avg50[sel])
im = ax[0].pcolormesh(eb, rb, H, cmap="Blues", shading="flat", edgecolors="#fcfcfb", linewidth=1)
cb = fig.colorbar(im, ax=ax[0], fraction=0.05)
cb.set_label("median avg. GDP growth 2025–50 (%/yr)", color=INK2)
cb.outline.set_visible(False)
ax[0].grid(False)
ax[0].set_xlabel("ε  (elasticity of substitution between tasks)")
ax[0].set_ylabel("r  (returns to AI-software research)")
ax[0].set_title("Growth by ε and r (other parameters random)", loc="left")

# (b) distribution
bins = np.linspace(0, 40, 41)
ax[1].hist(np.clip(100 * avg50, 0, 40), bins=bins, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.5,
           label="2025–2050")
ax[1].hist(np.clip(100 * avg75, 0, 40), bins=bins, color="#eb6834", alpha=0.55, edgecolor="#fcfcfb",
           linewidth=1.5, label="2050–2075")
txt = []
for arr, per in [(avg50, "2025–50"), (avg75, "2050–75")]:
    p10, p50, p90 = np.percentile(100 * arr, [10, 50, 90])
    txt.append(f"{per}: median {p50:.1f}%, p10–p90 {p10:.1f}–{p90:.1f}%")
ax[1].text(0.98, 0.70, "\n".join(txt), transform=ax[1].transAxes, ha="right", va="top",
           fontsize=8.5, color=INK2)
ax[1].set_xlabel("average annual GDP growth (%; values >40% binned at 40)")
ax[1].set_ylabel("number of simulations")
ax[1].set_title(f"Distribution over {len(avg50):,} draws", loc="left")
ax[1].legend(frameon=False)

# (c) sensitivity: Spearman rank correlation
labels = {"kappa_mid": "Capability difficulty κ_mid", "eps": "Task elasticity ε",
          "r": "Software returns r", "tau": "Adoption lag τ", "g_P": "Power build-out g_P",
          "beta_max": "Max automation β_max", "m_max": "Lab spending m_max"}
names = list(labels)
c50 = [spearmanr(draws[k], avg50).correlation for k in names]
c75 = [spearmanr(draws[k], avg75).correlation for k in names]
order = np.argsort(np.abs(c50))
yy = np.arange(len(names))
ax[2].barh(yy + 0.2, np.array(c50)[order], height=0.38, color="#2a78d6", label="growth 2025–50")
ax[2].barh(yy - 0.2, np.array(c75)[order], height=0.38, color="#eb6834", label="growth 2050–75")
ax[2].set_yticks(yy, [labels[names[k]] for k in order])
ax[2].axvline(0, color=INK2, lw=0.8)
ax[2].set_xlabel("Spearman rank correlation with growth")
ax[2].set_title("What drives the outcome?", loc="left")
ax[2].legend(frameon=False, loc="lower right")
ax[2].grid(axis="y", visible=False)

fig.tight_layout()
fig.savefig(P("fig2_monte_carlo.png"), dpi=160)

lines = []
lines.append("ENDO_AI v1.0 - Monte Carlo summary (N = %d draws)" % len(avg50))
lines.append("")
lines.append("Average annual GDP growth 2025-2050, percentiles 5/25/50/75/95 (%%): %s"
             % np.round(100 * np.percentile(avg50, [5, 25, 50, 75, 95]), 1).tolist())
lines.append("Average annual GDP growth 2050-2075, percentiles 5/25/50/75/95 (%%): %s"
             % np.round(100 * np.percentile(avg75, [5, 25, 50, 75, 95]), 1).tolist())
lines.append("Share of draws with average growth > 10%% in 2025-2050: %.3f" % (avg50 > 0.10).mean())
lines.append("Labour share 2050, percentiles 5/50/95 (%%): %s"
             % np.round(100 * np.percentile(ls50, [5, 50, 95]), 1).tolist())
lines.append("")
lines.append("Spearman rank correlation of parameters with average growth:")
lines.append("%-12s %10s %10s" % ("parameter", "2025-50", "2050-75"))
for k in names:
    lines.append("%-12s %+10.2f %+10.2f" % (k, spearmanr(draws[k], avg50).correlation,
                                            spearmanr(draws[k], avg75).correlation))
lines.append("")
lines.append("Scenario summary:")
for name, o in runs.items():
    def at(yr, k):
        return o[k][np.argmin(abs(yrs - yr)), 0]
    lines.append("%-15s growth 2030=%.1f%% 2050=%.1f%% 2070=%.1f%% | automated 2050=%.0f%% | "
                 "wage index 2050=%.3g | labour share 2050=%.0f%%"
                 % (name, 100 * at(2030, "g"), 100 * at(2050, "g"), 100 * at(2070, "g"),
                    100 * at(2050, "beta"), at(2050, "w") / o["w"][0, 0], 100 * at(2050, "lab_share")))
text = "\n".join(lines)
print(text)
with open(P("summary.txt"), "w") as fh:
    fh.write(text + "\n")

# one row per Monte Carlo draw
with open(P("monte_carlo_draws.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    pk = list(draws)
    wr.writerow(pk + ["avg_growth_2025_50", "avg_growth_2050_75", "labour_share_2050"])
    for j in range(len(avg50)):
        wr.writerow([f"{draws[k][j]:.6g}" for k in pk] + [f"{avg50[j]:.6g}", f"{avg75[j]:.6g}", f"{ls50[j]:.6g}"])
