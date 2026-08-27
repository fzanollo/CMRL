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
    gain_per_instance.csv    per-instance gain (aggregated across runs)
    gains_summary.tex        LaTeX table (mean/median gain, win rate)
    gains_distribution.tex   LaTeX table (P5/Q1/median/Q3/P95, IQR, skewness)
    gains_distribution.csv   same, plus min/max and heavy-tail instance counts
    boxplot_execution_time.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import config
from analysis.data_loading import load_all_runs
from analysis.gains_core import compute_agent_gains, distribution_summary, win_rate_summary


def build_summary_table(per_instance: pd.DataFrame, agent_keys: List[str]) -> pd.DataFrame:
    rows = []
    for agent_key in agent_keys:
        stats = win_rate_summary(per_instance.loc[per_instance["agent"] == agent_key, "gain_pct"])
        rows.append({"agent": agent_key, "label": config.HEURISTICS[agent_key]["label"], **stats})
    return pd.DataFrame(rows)


def build_distribution_table(per_instance: pd.DataFrame, agent_keys: List[str]) -> pd.DataFrame:
    rows = []
    for agent_key in agent_keys:
        stats = distribution_summary(per_instance.loc[per_instance["agent"] == agent_key, "gain_pct"])
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


def write_latex_distribution_table(dist: pd.DataFrame, baseline_key: str, out_path: Path) -> None:
    """Compact companion table: quartiles / P5 / P95 / IQR / skewness of the gains."""
    baseline_label = config.HEURISTICS[baseline_key]["label"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\caption{{Distribution of the per-instance execution-time gain (\\%) of CMRL over "
        f"{baseline_label}. The gain is bounded above by $+100\\%$ but unbounded below, so the "
        f"distribution is strongly left-skewed and the median is the representative summary; "
        f"the negative mean in Table~\\ref{{tab:cmrl_gains}} is driven by a small heavy lower tail.}}",
        r"\label{tab:cmrl_gains_dist}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"\textbf{CMRL} & \textbf{P5} & \textbf{Q1} & \textbf{Median} & \textbf{Q3} "
        r"& \textbf{P95} & \textbf{IQR} & \textbf{Skew} \\",
        r"\hline",
    ]
    for _, row in dist.iterrows():
        if row["n"] == 0:
            lines.append(f"{row['label']} & N/A & N/A & N/A & N/A & N/A & N/A & N/A \\\\")
            continue
        lines.append(
            f"{row['label']} & {row['p5']:+.1f} & {row['q1']:+.1f} & \\textbf{{{row['median']:+.1f}}} "
            f"& {row['q3']:+.1f} & {row['p95']:+.1f} & {row['iqr']:.1f} & {row['skew']:.1f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_gain_boxplot(per_instance: pd.DataFrame, agent_keys: List[str], out_path: Path) -> None:
    labels = [config.HEURISTICS[k]["label"] for k in agent_keys]
    colors = [config.HEURISTICS[k]["color"] for k in agent_keys]
    data = [per_instance.loc[per_instance["agent"] == k, "gain_pct"].dropna().values for k in agent_keys]

    box_half_width = 0.22
    n = len(data)
    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showfliers=False,
        widths=2 * box_half_width,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Annotate each box with its five-number-summary values so the figure carries
    # numbers, not just visual position. Labels sit clear of the box in the empty
    # margin -- the first box's labels go left, the last box's go right -- each
    # with a thin leader line back to the value. The lower whisker cap is the
    # 1.5*IQR fence; instances beyond it are the heavy lower tail quantified in
    # Table~\ref{tab:cmrl_gains_dist} (explained in the caption).
    ax.set_xlim(0.15, n + 0.85)
    for i, d in enumerate(data):
        if len(d) == 0:
            continue
        x = i + 1
        median_val = bp["medians"][i].get_ydata()[0]
        q1_val, q3_val = np.percentile(d, [25, 75])
        lower_cap = bp["caps"][2 * i].get_ydata()[0]

        to_right = i >= n - 1  # last box -> labels on the right, others on the left
        x_anchor = x + box_half_width if to_right else x - box_half_width
        dx = 16 if to_right else -16
        leader = dict(
            textcoords="offset points", fontsize=12,
            ha="left" if to_right else "right", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45", shrinkA=0, shrinkB=1),
        )
        ax.annotate(f"Q3 {q3_val:+.0f}%", xy=(x_anchor, q3_val), xytext=(dx, 4), **leader)
        # nudge the median label off the y=0 dashed line it sits next to
        ax.annotate(f"median {median_val:+.1f}%", xy=(x_anchor, median_val), xytext=(dx, -11), **leader)
        ax.annotate(f"Q1 {q1_val:+.0f}%", xy=(x_anchor, q1_val), xytext=(dx, -4), **leader)
        ax.annotate(f"whisker {lower_cap:+.0f}%", xy=(x, lower_cap), xytext=(dx, -9), **leader)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Execution time gain (%)  (positive = faster)", fontsize=12)
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

    distribution = build_distribution_table(per_instance, args.agents)
    distribution.to_csv(args.output_dir / "gains_distribution.csv", index=False)
    write_latex_distribution_table(distribution, args.baseline, args.output_dir / "gains_distribution.tex")

    plot_gain_boxplot(per_instance, args.agents, args.output_dir / "boxplot_execution_time.png")

    print("\nSUMMARY (median gain, win rate vs. {}):".format(config.HEURISTICS[args.baseline]["label"]))
    print(summary[["label", "n", "mean_gain", "median_gain", "improved", "win_rate"]].to_string(index=False))
    print("\nGAIN DISTRIBUTION (per-instance gain %):")
    print(distribution[["label", "n", "min", "p5", "q1", "median", "q3", "p95",
                        "iqr", "skew", "n_below_100", "n_below_200"]].to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
