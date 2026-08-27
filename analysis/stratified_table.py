#!/usr/bin/env python3
"""Per-stratum (per grid size) win-rate table, e.g. the paper's
"Stratified win rates of CMRL against the baseline RL heuristic" table.

Same gain computation as gains_table.py, just grouped by `problem` instead
of pooled across all of them.

    python -m analysis.stratified_table
    python -m analysis.stratified_table --baseline ra --agents cmrl_400e cmrl_1000e
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from analysis import config
from analysis.data_loading import load_all_runs
from analysis.gains_core import compute_agent_gains, win_rate_summary


def build_stratified_table(per_instance: pd.DataFrame, problems: List[str], agent_keys: List[str]) -> pd.DataFrame:
    rows = []
    for problem in problems:
        for agent_key in agent_keys:
            subset = per_instance[(per_instance["problem"] == problem) & (per_instance["agent"] == agent_key)]
            stats = win_rate_summary(subset["gain_pct"])
            rows.append({
                "problem": problem,
                "agent": agent_key,
                "label": config.HEURISTICS[agent_key]["label"],
                **stats,
            })
    return pd.DataFrame(rows)


def write_latex_table(stratified: pd.DataFrame, problems: List[str], agent_keys: List[str], out_path: Path) -> None:
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Stratified win rates (\%) of CMRL against the baseline RL heuristic, broken down by stratum and training duration.}",
        r"\label{tab:win-rates-by-size}",
        r"\begin{tabular}{ll" + "r" * len(agent_keys) + "}",
        r"\toprule",
        r"\textbf{Problem} & \textbf{Method} & \textbf{Win Rate (\%)} \\",
        r"\midrule",
    ]
    for i, problem in enumerate(problems):
        lines.append(f"\\multirow{{{len(agent_keys)}}}{{*}}{{{problem}}} ")
        for agent_key in agent_keys:
            row = stratified[(stratified["problem"] == problem) & (stratified["agent"] == agent_key)].iloc[0]
            win_txt = f"{row['win_rate']:.1f}" if row["n"] else "N/A"
            lines.append(f"    & {row['label']} & {win_txt} \\\\")
        if i < len(problems) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=config.DEFAULT_RESULTS_ROOT)
    parser.add_argument("--problems", nargs="+", default=config.DEFAULT_PROBLEMS)
    parser.add_argument("--runs", type=int, default=config.DEFAULT_RUNS)
    parser.add_argument("--metric", default=config.DEFAULT_METRIC, choices=config.METRICS)
    parser.add_argument("--baseline", default=config.DEFAULT_BASELINE, choices=list(config.HEURISTICS))
    parser.add_argument("--agents", nargs="+", default=config.DEFAULT_AGENTS, choices=list(config.HEURISTICS))
    parser.add_argument("--instance-agg", default="median", choices=["mean", "median"])
    parser.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_runs(args.results_root, args.problems, args.runs)

    print(f"Computing per-stratum gains: baseline={args.baseline!r} vs agents={args.agents}...")
    per_instance = compute_agent_gains(df, args.baseline, args.agents, args.metric, args.instance_agg)

    stratified = build_stratified_table(per_instance, args.problems, args.agents)
    stratified.to_csv(args.output_dir / "winrate_stratified.csv", index=False)
    write_latex_table(stratified, args.problems, args.agents, args.output_dir / "winrate_stratified.tex")

    print("\nWIN RATE BY STRATUM:")
    print(stratified[["problem", "label", "n", "win_rate"]].to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
