# ENDO_AI v1.0

An endogenous growth model with AI: task-based automation, AI-automated research, recursive AI-software progress, and limits from compute and power. Version 1 uses rules for saving and allocation instead of an intertemporal equilibrium.

| Folder | Contents |
|---|---|
| `src/` | `ai_growth_model.py` (the model) and `run_and_plot.py` (scenarios, Monte Carlo, figures) |
| `doc/` | [`MODEL.md`](doc/MODEL.md): full documentation with equations, calibration, results and limitations |
| `output/` | All results: figures, scenario paths, Monte Carlo draws, summary |

```bash
pip install -r requirements.txt
python src/run_and_plot.py   # regenerates everything in output/
```

## Output files

| File | Contents |
|---|---|
| `fig1_scenarios.png` | GDP growth, automation, wages and labour share for the three scenarios |
| `fig2_monte_carlo.png` | Growth by ε and r, distribution of outcomes, parameter sensitivity |
| `scenario_paths.csv` | Yearly paths of the main variables, by scenario |
| `monte_carlo_draws.csv` | One row per draw: parameters and outcomes (6,000 rows) |
| `summary.txt` | Summary statistics and sensitivity table |
