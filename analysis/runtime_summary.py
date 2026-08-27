#!/usr/bin/env python3
"""Total and per-instance execution-time summary across heuristics.

Produces the numbers behind claims like "RA is ~4x slower per instance than
the RL-based heuristics": per-instance mean/median execution time for each
heuristic, plus its ratio to a chosen reference heuristic, and the total
wall-clock time each heuristic would need across the whole benchmark
(median +/- std across the 5 training/testing runs; RA and other
training-free heuristics are measured once, in run_1).

    python -m analysis.runtime_summary
    python -m analysis.runtime_summary --heuristics ra prev_rl cmrl_400e --reference cmrl_400e
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from analysis import config
from analysis.data_loading import load_all_runs, filter_heuristics


def per_instance_stats(df: pd.DataFrame, heuristic_keys: List[str], metric: str) -> pd.DataFrame:
    rows = []
    for key in heuristic_keys:
        model = config.HEURISTICS[key]["model"]
        vals = df.loc[df["Model"] == model, metric].dropna()
        rows.append({
            "heuristic": key,
            "label": config.HEURISTICS[key]["label"],
            "n": len(vals),
            "mean": vals.mean() if len(vals) else float("nan"),
            "median": vals.median() if len(vals) else float("nan"),
        })
    stats = pd.DataFrame(rows)
    return stats


def add_ratios(stats: pd.DataFrame, reference_key: str) -> pd.DataFrame:
    ref_row = stats.loc[stats["heuristic"] == reference_key].iloc[0]
    stats = stats.copy()
    stats["mean_ratio_vs_ref"] = stats["mean"] / ref_row["mean"]
    stats["median_ratio_vs_ref"] = stats["median"] / ref_row["median"]
    return stats


def total_runtime_per_run(df: pd.DataFrame, heuristic_keys: List[str], metric: str) -> pd.DataFrame:
    """Sum of `metric` across all instances, per (heuristic, run_id)."""
    rows = []
    for key in heuristic_keys:
        model = config.HEURISTICS[key]["model"]
        subset = df[df["Model"] == model]
        totals = subset.groupby("run_id")[metric].sum()
        for run_id, total in totals.items():
            rows.append({"heuristic": key, "label": config.HEURISTICS[key]["label"], "run_id": run_id, "total": total})
    return pd.DataFrame(rows)


def summarize_totals(totals: pd.DataFrame) -> pd.DataFrame:
    summary = (
        totals.groupby(["heuristic", "label"])["total"]
        .agg(median="median", mean="mean", std="std", n_runs="count")
        .reset_index()
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=config.DEFAULT_RESULTS_ROOT)
    parser.add_argument("--problems", nargs="+", default=config.DEFAULT_PROBLEMS)
    parser.add_argument("--runs", type=int, default=config.DEFAULT_RUNS)
    parser.add_argument("--metric", default=config.DEFAULT_METRIC, choices=config.METRICS)
    parser.add_argument(
        "--heuristics", nargs="+", default=config.DEFAULT_TABLE_HEURISTICS, choices=list(config.HEURISTICS),
        help="Heuristics to include (default: %(default)s).",
    )
    parser.add_argument(
        "--reference", default=config.DEFAULT_BASELINE, choices=list(config.HEURISTICS),
        help="Heuristic used as the denominator for the ratio columns (default: %(default)s).",
    )
    parser.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_runs(args.results_root, args.problems, args.runs)
    df = filter_heuristics(df, args.heuristics)

    stats = per_instance_stats(df, args.heuristics, args.metric)
    stats = add_ratios(stats, args.reference)
    stats.to_csv(args.output_dir / "runtime_per_instance_summary.csv", index=False)

    totals = total_runtime_per_run(df, args.heuristics, args.metric)
    totals_summary = summarize_totals(totals)
    totals.to_csv(args.output_dir / "runtime_totals_per_run.csv", index=False)
    totals_summary.to_csv(args.output_dir / "runtime_totals_summary.csv", index=False)

    print(f"\nPER-INSTANCE {args.metric} (mean/median, ratio vs. {config.HEURISTICS[args.reference]['label']}):")
    print(stats.to_string(index=False))

    print(f"\nTOTAL {args.metric} ACROSS ALL INSTANCES (median +/- std across runs):")
    print(totals_summary.to_string(index=False))

    print(f"\nSaved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
