#!/usr/bin/env python3
"""Training-time comparison plot: CMRL (monolithic synthesis + DQN training)
vs. the prior interleaved RL baseline's single training run.

Reproduces the paper's "Total training time comparison" figures
(training_time_comparison_400e.png / _1000e.png).

    python -m analysis.training_time_plot
    python -m analysis.training_time_plot --episodes 400 1000 --run 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import config


def load_prev_rl_training_time(run_dir: Path) -> float:
    matches = list((run_dir / "trained_agents" / "prev_rl").glob("training_times_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No prev_rl training-time file under {run_dir / 'trained_agents' / 'prev_rl'}")
    with open(matches[0], "r", encoding="utf-8") as f:
        first_line = f.readline()
    return float(first_line.split(":")[1].strip())


def load_dqn_training_time(run_dir: Path, problem: str, episodes: int) -> float:
    agent_name = f"{problem}_{episodes}e_dqn_agent"
    matches = list((run_dir / "trained_agents" / agent_name / "training_results").glob("*_training_info.csv"))
    if not matches:
        raise FileNotFoundError(f"No training_info.csv for {agent_name} under {run_dir}")
    info = pd.read_csv(matches[0])
    return float(info["training_time_seconds"].iloc[0])


def load_monolithic_time(monolithic_dir: Path, problem: str) -> float:
    csv_path = monolithic_dir / problem / "monolithic_results.csv"
    return float(pd.read_csv(csv_path)["execution_time"].iloc[0])


def make_plot(
    problems: List[str],
    prev_rl_time: float,
    dqn_times: List[float],
    monolithic_times: List[float],
    episodes: int,
    out_path: Path,
) -> None:
    x = np.arange(len(problems))
    width = 0.4
    total_times = [d + m for d, m in zip(dqn_times, monolithic_times)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, dqn_times, width, label="DQN Training Time", color="tab:blue")
    ax.bar(x, monolithic_times, width, bottom=dqn_times, label="Monolithic Solving Time", color="tab:green")
    ax.plot(x, total_times, marker="o", linewidth=2, markersize=8, color="tab:purple",
            label="CMRL = DQN + Monolithic", zorder=5)
    ax.axhline(y=prev_rl_time, color="red", linestyle="--", linewidth=2,
               label=f"Previous RL Training Time ({prev_rl_time:.1f}s)", zorder=3)

    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Training Times Comparison: DQN+Monolithic vs Previous RL ({episodes} episodes)")
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=45, ha="right")
    ax.legend(loc="best")

    for xi, yi in zip(x, total_times):
        ax.annotate(f"{yi:.1f}", xy=(xi, yi), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=config.DEFAULT_RESULTS_ROOT)
    parser.add_argument("--monolithic-dir", type=Path, default=config.DEFAULT_MONOLITHIC_DIR)
    parser.add_argument("--problems", nargs="+", default=config.DEFAULT_PROBLEMS)
    parser.add_argument("--episodes", nargs="+", type=int, default=[400, 1000])
    parser.add_argument("--run", type=int, default=1, help="Which run_N to read training times from.")
    parser.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.results_root / f"run_{args.run}"

    prev_rl_time = load_prev_rl_training_time(run_dir)

    for episodes in args.episodes:
        dqn_times = [load_dqn_training_time(run_dir, p, episodes) for p in args.problems]
        monolithic_times = [load_monolithic_time(args.monolithic_dir, p) for p in args.problems]

        out_path = args.output_dir / f"training_time_comparison_{episodes}e.png"
        make_plot(args.problems, prev_rl_time, dqn_times, monolithic_times, episodes, out_path)
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
