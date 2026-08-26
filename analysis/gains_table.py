#!/usr/bin/env python3
"""Compute gain/win-rate of one or more agents against a baseline heuristic,
and export a LaTeX summary table plus a gain boxplot.

This is the script behind the paper's main comparison table (Table:
"Performance of CMRL compared to previous RL approach"). By default it
reproduces that exact comparison (baseline=Previous RL, agents=CMRL 400e/1000e),
but any heuristic registered in analysis/config.py can be used as baseline or
agent -- e.g. to compare CMRL against Ready Abstraction instead:

    python -m analysis.gains_table --baseline ra --agents cmrl_400e cmrl_1000e

Outputs (under --output-dir):
    gain_per_instance.csv   per-instance gain (aggregated across runs)
    gains_summary.tex       LaTeX table (mean/median gain, win rate)
    boxplot_execution_time.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .data_loading import load_all_runs
from .gains_core import compute_agent_gains, win_rate_summary


def build_summary_table(per_instance: pd.DataFrame, agent_keys: List[str]) -> pd.DataFrame:
    rows = []
    for agent_key in agent_keys:
        stats = win_rate_summary(per_instance.loc[per_instance["agent"] == agent_key, "gain_pct"])
        rows.append({"agent": agent_key, "label": config.HEURISTICS[agent_key]["label"], **stats})
    return pd.DataFrame(rows)


def write_latex_table(summary: pd.DataFrame, baseline_key: str, out_path: Path) -> None:
    baseline_label = config.HEURISTICS[baseline_key]["label"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Performance of CMRL compared to {baseline_label} across all selected instances.}}",
        r"\label{tab:cmrl_gains}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"\textbf{CMRL} & \textbf{Mean Gain} & \textbf{Median Gain} & \textbf{Improved} & \textbf{Win Rate} \\",
        r"\hline",
    ]
    for _, row in summary.iterrows():
        if row["n"] == 0:
            lines.append(f"{row['label']} & N/A & N/A & 0/0 & N/A \\\\")
            continue
        lines.append(
            f"{row['label']} & {row['mean_gain']:+.1f}\\% & \\textbf{{{row['median_gain']:+.1f}\\%}} "
            f"& {row['improved']}/{row['n']} & \\textbf{{{row['win_rate']:.1f}\\%}} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_gain_boxplot(per_instance: pd.DataFrame, agent_keys: List[str], out_path: Path) -> None:
    labels = [config.HEURISTICS[k]["label"] for k in agent_keys]
    colors = [config.HEURISTICS[k]["color"] for k in agent_keys]
    data = [per_instance.loc[per_instance["agent"] == k, "gain_pct"].dropna().values for k in agent_keys]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for i, line in enumerate(bp["medians"]):
        median_val = line.get_ydata()[0]
        if pd.notna(median_val):
            ax.text(i + 1, median_val, f" {median_val:+.1f}%", va="center", fontsize=9)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Execution time gain (%)  (positive = better)")
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=config.DEFAULT_RESULTS_ROOT)
    parser.add_argument("--problems", nargs="+", default=config.DEFAULT_PROBLEMS)
    parser.add_argument("--runs", type=int, default=config.DEFAULT_RUNS)
    parser.add_argument("--metric", default=config.DEFAULT_METRIC, choices=config.METRICS)
    parser.add_argument(
        "--baseline", default=config.DEFAULT_BASELINE, choices=list(config.HEURISTICS),
        help="Heuristic used as the reference in the gain formula (default: %(default)s).",
    )
    parser.add_argument(
        "--agents", nargs="+", default=config.DEFAULT_AGENTS, choices=list(config.HEURISTICS),
        help="Heuristic(s) compared against the baseline (default: %(default)s).",
    )
    parser.add_argument(
        "--instance-agg", default="median", choices=["mean", "median"],
        help="How to collapse per-run measurements into one value per instance before computing win rate.",
    )
    parser.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_runs(args.results_root, args.problems, args.runs)

    print(f"Computing gains: baseline={args.baseline!r} vs agents={args.agents} (metric={args.metric!r})...")
    per_instance = compute_agent_gains(df, args.baseline, args.agents, args.metric, args.instance_agg)
    per_instance.to_csv(args.output_dir / "gain_per_instance.csv", index=False)

    summary = build_summary_table(per_instance, args.agents)
    write_latex_table(summary, args.baseline, args.output_dir / "gains_summary.tex")
    plot_gain_boxplot(per_instance, args.agents, args.output_dir / "boxplot_execution_time.png")

    print("\nSUMMARY (median gain, win rate vs. {}):".format(config.HEURISTICS[args.baseline]["label"]))
    print(summary[["label", "n", "mean_gain", "median_gain", "improved", "win_rate"]].to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
