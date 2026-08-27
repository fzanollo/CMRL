"""Shared gain/win-rate computation used by gains_table.py and stratified_table.py.

Gain formula (lower metric values are better, e.g. execution_time):

    gain_pct = (baseline_value - agent_value) / baseline_value * 100

A positive gain means the agent is faster/better than the baseline on that
instance. Values are first paired per (problem, instance_name, run_id) so
that an agent is only ever compared against the baseline measured in the
*same* run -- this is what makes the pairing well defined even when the
baseline (e.g. RA) was only measured once, in run_1: the inner join simply
restricts the comparison to that run.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from . import config


def pairwise_gains(df: pd.DataFrame, baseline_model: str, agent_model: str, metric: str) -> pd.DataFrame:
    """Per-instance, per-run gain of `agent_model` over `baseline_model` for `metric`."""
    key = ["problem", "instance_name", "run_id"]
    baseline = df[df["Model"] == baseline_model][[*key, metric]].rename(columns={metric: "baseline_value"})
    agent = df[df["Model"] == agent_model][[*key, metric]].rename(columns={metric: "agent_value"})

    merged = baseline.merge(agent, on=key, how="inner")
    if merged.empty:
        return merged.assign(gain_pct=pd.Series(dtype=float))

    merged["gain_pct"] = (
        (merged["baseline_value"] - merged["agent_value"]) / merged["baseline_value"].replace(0, np.nan) * 100
    ).replace([np.inf, -np.inf], np.nan)
    return merged


def _mad(x: pd.Series) -> float:
    """Median absolute deviation."""
    return (x - x.median()).abs().median()


def aggregate_across_runs(gains_df: pd.DataFrame, instance_agg: str) -> pd.DataFrame:
    """Collapse per-run gains to one value per (problem, instance_name) using mean or median."""
    if instance_agg not in ("mean", "median"):
        raise ValueError("instance_agg must be 'mean' or 'median'")

    agg = (
        gains_df.groupby(["problem", "instance_name"])["gain_pct"]
        .agg(gain_pct=instance_agg, mad=_mad, n_runs="count")
        .reset_index()
    )
    return agg


def win_rate_summary(per_instance_gains: pd.Series) -> dict:
    """Mean/median gain and win rate over a series of per-instance gains."""
    vals = per_instance_gains.dropna()
    n = len(vals)
    improved = int((vals > 0).sum()) if n else 0
    return {
        "n": n,
        "mean_gain": float(vals.mean()) if n else float("nan"),
        "median_gain": float(vals.median()) if n else float("nan"),
        "improved": improved,
        "win_rate": (improved / n * 100.0) if n else float("nan"),
    }


def distribution_summary(per_instance_gains: pd.Series) -> dict:
    """Distributional characterization of a per-instance gain series.

    The gain metric is bounded above by +100% (an agent can at best be
    instantaneous) but unbounded below, so the distribution is typically
    strongly left-skewed and the median is more representative than the mean.
    This returns the quartiles, the 5th/95th percentiles, the inter-quartile
    range, the (adjusted Fisher-Pearson) skewness, and counts of instances in
    the heavy lower tail.
    """
    vals = per_instance_gains.dropna()
    n = len(vals)
    if n == 0:
        keys = ("min", "p5", "q1", "median", "q3", "p95", "max", "iqr", "skew")
        return {"n": 0, **{k: float("nan") for k in keys},
                "n_below_50": 0, "n_below_100": 0, "n_below_200": 0}

    p5, q1, median, q3, p95 = vals.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": n,
        "min": float(vals.min()),
        "p5": float(p5),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "p95": float(p95),
        "max": float(vals.max()),
        "iqr": float(q3 - q1),
        "skew": float(vals.skew()),
        "n_below_50": int((vals < -50).sum()),
        "n_below_100": int((vals < -100).sum()),
        "n_below_200": int((vals < -200).sum()),
    }


def compute_agent_gains(
    df: pd.DataFrame,
    baseline_key: str,
    agent_keys: List[str],
    metric: str = config.DEFAULT_METRIC,
    instance_agg: str = "median",
) -> pd.DataFrame:
    """Per-instance gains of each agent vs. the baseline, aggregated across runs.

    Returns one row per (problem, instance_name, agent) with columns
    gain_pct, mad, n_runs.
    """
    heuristics = config.resolve_heuristics([baseline_key, *agent_keys])
    baseline_model = heuristics[baseline_key]["model"]

    frames = []
    for agent_key in agent_keys:
        agent_model = heuristics[agent_key]["model"]
        raw = pairwise_gains(df, baseline_model, agent_model, metric)
        if raw.empty:
            print(f"[gains_core] No overlapping instances for baseline={baseline_key!r} vs agent={agent_key!r}.")
            continue
        agg = aggregate_across_runs(raw, instance_agg)
        agg["agent"] = agent_key
        frames.append(agg)

    if not frames:
        return pd.DataFrame(columns=["problem", "instance_name", "agent", "gain_pct", "mad", "n_runs"])
    return pd.concat(frames, ignore_index=True)
