# ENDO_AI v2.0

Perfect-foresight general equilibrium version of the ENDO_AI model. The technology is the same as in [v1.0](../v1.0/): task-based automation, AI-automated research, recursive AI-software progress, and compute and power constraints. v1.0's behavioural rules are replaced by forward-looking agents:

- **Ramsey households:** the interest rate satisfies the Euler equation.
- **Compute owners:** invest until the compute rental covers the user cost, including expected capital losses from falling chip prices.
- **AI labs:** price at a markup and choose their R&D compute to maximise the present value of profits.

| Folder | Contents |
|---|---|
| `src/` | `endo_ai_v2.py` (model and solver), `run_v2.py` (scenarios, comparison with v1.0, figures, summary), `run_mc_v2.py` (Monte Carlo) |
| `doc/` | [`MODEL.md`](doc/MODEL.md): equations, equilibrium definition, solution algorithm, calibration, results and interpretation |
| `output/` | All results (see below) |

```bash
pip install -r requirements.txt
python src/run_mc_v2.py 100   # Monte Carlo (slow), writes output/mc_cache.pkl
python src/run_v2.py          # scenarios, figures, CSVs, summary
```

## Output files

| File | Contents |
|---|---|
| `fig1_equilibrium_paths.png` | Growth, interest rate, saving rate, AI investment, lab R&D share and automation for the three scenarios |
| `fig2_v1_vs_v2.png` | GDP growth and AI investment: v2.0 compared with v1.0 |
| `fig3_wages_consumption.png` | Real wage, consumption per head and labour share |
| `fig4_monte_carlo.png` | Distribution of growth, early AI investment vs later growth, parameter sensitivity |
| `scenario_paths.csv` | Yearly equilibrium paths 2025–2075, by scenario |
| `monte_carlo_draws.csv` | One row per draw: parameters, outcomes, convergence diagnostics |
| `summary.txt` | Key results, comparison with v1.0, convergence diagnostics, Monte Carlo statistics |
| `scenario_cache.pkl`, `mc_cache.pkl` | Solved equilibria (Python pickle), so figures can be regenerated without re-solving |
