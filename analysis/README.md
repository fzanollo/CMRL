# Analysis & Table Generation

Scripts that turn the raw `run_mtsa.py` output (`rl_results/run_*/results/**/comparative_results_all.csv`,
`ra_results/`, `monolithic_results/`) into the summary tables, LaTeX snippets, and plots used in the paper.

Gain/win-rate computation against any baseline heuristic is centralized in `gains_core.py` and shared by
`gains_table.py` and `stratified_table.py`, so comparisons (e.g. "vs Previous RL" or "vs RA") are just a
`--baseline` flag rather than separate scripts.

## Layout

- `config.py` — all defaults live here: paths, the list of Sokoban problems, and the **heuristic registry**
  (`HEURISTICS`), which maps a short CLI key to the exact `Model` value in `comparative_results_all.csv`
  plus its display label/color/marker.
- `data_loading.py` — loads and concatenates `comparative_results_all.csv` across runs/problems into one
  tidy dataframe; filters to `result == "OK"` by default.
- `gains_core.py` — shared gain/win-rate computation (pairwise per-run gain → aggregate across runs →
  win-rate summary). Used by both table scripts below.
- `gains_table.py` — the paper's main comparison table (mean/median gain, win rate) + boxplot.
- `stratified_table.py` — the same comparison broken down per grid size (stratum).
- `runtime_summary.py` — total and per-instance execution-time summary across heuristics, with ratios to a
  reference heuristic (this is what backs claims like "RA is ~4x slower per instance").
- `training_time_plot.py` — training-cost comparison figure (CMRL's monolithic + DQN time vs. the prior
  approach's single training run).

## Heuristics option

Every table/plot script accepts `--heuristics` / `--baseline` / `--agents` (all keys from `config.HEURISTICS`).
The defaults cover the three families compared in the paper (CMRL has two training-length variants):

| key          | Model column value | label             |
|--------------|---------------------|-------------------|
| `ra`         | `RA`                | Ready Abstraction |
| `prev_rl`    | `Previous RL`       | Delgado et al.    |
| `cmrl_400e`  | `DQN Agent 400e`    | CMRL 400e         |
| `cmrl_1000e` | `DQN Agent 1000e`   | CMRL 1000e        |

`DEFAULT_BASELINE = "prev_rl"` and `DEFAULT_AGENTS = ["cmrl_400e", "cmrl_1000e"]` reproduce the paper's
main table without passing any flags. To add a new heuristic (e.g. the graph-RL baseline), register it once
in `config.HEURISTICS` and every script below picks it up automatically.

## Usage

Run as modules from the repo root (so the relative imports resolve):

```bash
# Main gain/win-rate table + boxplot, CMRL vs Previous RL (paper default)
python -m analysis.gains_table

# Same, but against Ready Abstraction instead
python -m analysis.gains_table --baseline ra --agents cmrl_400e cmrl_1000e

# Per-stratum win-rate table
python -m analysis.stratified_table

# Total / per-instance execution time across heuristics, with ratios
python -m analysis.runtime_summary --reference prev_rl

# Training time comparison figure
python -m analysis.training_time_plot --episodes 400 1000
```

All scripts write to `analysis/output/` by default (`--output-dir` to change it); that folder is gitignored
since it's fully regenerable from the CSVs already in the repo.

## Notes

- Gain formula (lower metric is better, e.g. `execution_time`):
  `gain_pct = (baseline_value - agent_value) / baseline_value * 100`. Positive = the agent is faster.
- Pairing is always done per `(problem, instance_name, run_id)` — an agent is only compared against a
  baseline measurement from the *same* run. RA and other training-free heuristics were only measured once
  (in `run_1`), so any comparison involving them is automatically restricted to that run via the inner join;
  no special-casing needed.
- `data_loading.py` drops any row where `result != "OK"` (timeouts/errors) before computing gains, so a
  future run that includes failures won't silently corrupt the statistics.
