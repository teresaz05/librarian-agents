"""Summarize or plot the BrowseComp+ training reward used in Figure 7."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_points(path: Path) -> list[tuple[int, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = [(int(row["step"]), float(row["reward_total"])) for row in rows]
    if [step for step, _ in points] != list(range(len(points))):
        raise ValueError("Training steps must be consecutive and start at zero.")
    return points


def ema(values: list[float], smoothing: float) -> list[float]:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("EMA smoothing must be in [0, 1).")
    output: list[float] = []
    for value in values:
        output.append(value if not output else smoothing * output[-1] + (1 - smoothing) * value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/training_reward.csv"))
    parser.add_argument("--ema", type=float, default=0.9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    points = read_points(args.input)
    smoothed = ema([value for _, value in points], args.ema)
    print(f"points={len(points)} first={points[0][1]:.6f} last={points[-1][1]:.6f}")
    if args.output is None:
        return
    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4, 3.2))
    steps = [step for step, _ in points]
    values = [value for _, value in points]
    plt.scatter(steps, values, s=14, alpha=0.45, label="Reward")
    plt.plot(steps, smoothed, linewidth=2, label="EMA")
    plt.xlabel("Training step")
    plt.ylabel("Aggregate rollout reward")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
