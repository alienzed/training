#!/usr/bin/env python3
"""
Checkpoint Scoring Script — Region-Oriented Output
"""

import sys
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------
WINDOW = 5
ALPHA = 0.3
TREND_WINDOW = 8
TREND_WEIGHT = 0.4

REGIME_CONFIRM_STEPS = 3
REGIME_DROP_FRAC = 0.15
REGIME_MIN_STEP_FRAC = 0.15

DEPTH_GATE = 0.5

REGION_STEP_FRAC = 0.15     # regions must be ~15% of run apart
MAX_PER_REGION = 5          # epochs shown per region
MAX_REGIONS = 6             # max regions shown
# ------------------------------------------------


def main(csv_path: str):
    df = pd.read_csv(csv_path)

    steps = df["Step"].to_numpy()
    loss = df["Value"].to_numpy()

    max_step = steps.max()
    min_step = steps.min()
    run_span = max_step - min_step

    # ---- DEPTH ----
    depth_score = (loss.max() - loss) / (loss.max() - loss.min() + 1e-12)

    # ---- STABILITY ----
    stability_score = np.zeros_like(loss)
    for i in range(len(loss)):
        l = max(0, i - WINDOW)
        r = min(len(loss), i + WINDOW + 1)
        stability_score[i] = 1.0 / (1.0 + np.std(loss[l:r]))

    # ---- TREND ----
    trend_bonus = np.zeros_like(loss)
    for i in range(TREND_WINDOW, len(loss)):
        trend_bonus[i] = max(
            0.0,
            np.mean(loss[i - TREND_WINDOW:i]) - loss[i],
        )

    if trend_bonus.max() > 0:
        trend_bonus /= trend_bonus.max()

    # ==================================================
    # FIRST REAL REGIME DETECTION
    # ==================================================
    rolling_mean = (
        pd.Series(loss)
        .rolling(TREND_WINDOW)
        .mean()
        .to_numpy()
    )

    first_regime_step = None
    confirm = 0
    min_allowed = min_step + REGIME_MIN_STEP_FRAC * run_span

    for i in range(TREND_WINDOW, len(loss)):
        if steps[i] < min_allowed:
            continue

        drop = (
            (rolling_mean[i] - loss[i])
            / (rolling_mean[i] + 1e-12)
        )

        if drop > REGIME_DROP_FRAC:
            confirm += 1
            if confirm >= REGIME_CONFIRM_STEPS:
                first_regime_step = steps[i - REGIME_CONFIRM_STEPS + 1]
                break
        else:
            confirm = 0

    if first_regime_step is None:
        first_regime_step = steps[int(0.25 * len(steps))]

    valid_mask = steps >= first_regime_step

    # ---- EARLY WEIGHT (BASIN-RELATIVE) ----
    progress = (steps - first_regime_step) / (
        max_step - first_regime_step + 1e-12
    )
    progress = np.clip(progress, 0.0, 1.0)
    early_weight = (1.0 - progress) ** ALPHA

    # ---- GATES ----
    stability_score *= depth_score > DEPTH_GATE

    # ---- FINAL SCORE ----
    final_score = (
        depth_score
        * stability_score
        * early_weight
        * (1.0 + TREND_WEIGHT * trend_bonus)
    )

    final_score *= valid_mask

    df["final_score"] = final_score

    # ---- LOCAL MINIMA ----
    is_local_min = np.zeros(len(loss), dtype=bool)
    for i in range(1, len(loss) - 1):
        is_local_min[i] = loss[i] < loss[i - 1] and loss[i] < loss[i + 1]

    candidates = df[is_local_min & valid_mask].copy()
    candidates = candidates.sort_values("final_score", ascending=False)

    # ==================================================
    # REGION SELECTION
    # ==================================================
    region_min_dist = REGION_STEP_FRAC * run_span
    regions = []

    for _, row in candidates.iterrows():
        step = row["Step"]

        if len(regions) >= MAX_REGIONS:
            break

        if all(abs(step - r["center"]) >= region_min_dist for r in regions):
            regions.append({
                "center": step,
                "rows": [row],
            })
        else:
            for r in regions:
                if abs(step - r["center"]) < region_min_dist:
                    if len(r["rows"]) < MAX_PER_REGION:
                        r["rows"].append(row)
                    break

    # ---- OUTPUT ----
    print(f"\nDetected {len(regions)} good regions (from step ≥ {first_regime_step})\n")

    for i, region in enumerate(regions, 1):
        rows = pd.DataFrame(region["rows"]).sort_values(
            "final_score", ascending=False
        )

        print(f"Region {i} (around step ~{int(region['center'])}):")
        print(
            rows[
                ["Step", "Value", "final_score"]
            ].to_string(index=False)
        )
        print("")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python score-scalars.py <loss_log.csv>")
        sys.exit(1)

    main(sys.argv[1])
