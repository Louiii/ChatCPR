#!/usr/bin/env python3
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull, QhullError

# ------------------- CONFIG -------------------
# EXPS = "regular"
# EXPS = "saboteur"
EXPS = "saboteur_with_snitches"
# EXPS = "bad_kimi_rollouts"


SUFF = "" if EXPS == "regular" else f"_{EXPS}"
PLOTDIR = Path("plots/quantitative" + SUFF)
CSVDIR = Path(f"tables/{EXPS}")
METRICS_CSV = CSVDIR / "all_runs_metrics.csv"
OUT_PATH = PLOTDIR / "resource_vs_credits.mp4"

OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

# keep this mapping in sync with plots.py
runs = json.loads(Path("completed_experiments.json").read_text())[EXPS]


PLOTDIR.mkdir(exist_ok=True, parents=True)
sns.set_style("whitegrid")

df = pd.read_csv(METRICS_CSV)

# Map run_id -> experiment
run_to_exp = {}
for exp, run_list in runs.items():
    for r in run_list:
        run_to_exp[r] = exp
df["experiment"] = df["run_id"].map(run_to_exp)

# Drop runs that aren't mapped
df = df[df["experiment"].notna()].copy()

# Compute total credits safely (post if available else pre)
df["total_credits"] = np.where(
    df["credit_mean_post"].notna(),
    df["credit_mean_post"] * df["num_alive_post"],
    df["credit_mean_pre"] * df["num_alive_pre"],
)

# Insert round 0 if missing for any run
patched = []
for run_id, sub in df.groupby("run_id"):
    if 0 not in sub["round"].values:
        first = sub.iloc[0].copy()
        first["round"] = 0
        first["R_after_growth"] = first["R_before"]
        first["total_credits"] = first["credit_mean_pre"] * first["num_alive_pre"]
        patched.append(first)
if patched:
    df = pd.concat([df, pd.DataFrame(patched)], ignore_index=True)

df = df.sort_values(["experiment", "run_id", "round"])
rounds = sorted(df["round"].unique())
experiments = list(df["experiment"].unique())
palette = sns.color_palette("tab10", len(experiments))
exp_to_color = {exp: palette[i] for i, exp in enumerate(experiments)}

# Global axes limits with margins
xmax = df["R_after_growth"].max()
ymax = df["total_credits"].max()
xmargin = 0.05 * xmax if xmax > 0 else 1
ymargin = 0.05 * ymax if ymax > 0 else 1

fig, ax = plt.subplots(figsize=(8, 6))

# --- FIXED LEGEND OUTSIDE FIGURE ---
legend_handles = [
    plt.Line2D([], [], color=exp_to_color[exp], marker='o', linestyle='', markersize=8)
    for exp in experiments
]

fig.legend(
    legend_handles,
    experiments,
    title="",
    fontsize=8,
    loc="center right",
    bbox_to_anchor=(0.98, 0.5),
    borderaxespad=0.,
)

plt.subplots_adjust(right=0.78)



def draw_cov_ellipse(ax, xs, ys, color, alpha=0.07, n_std=1.5):
    """Fallback: draw covariance ellipse centered at mean(xs,ys)."""
    if len(xs) < 2:
        return
    x_m, y_m = np.mean(xs), np.mean(ys)
    cov = np.cov(np.vstack([xs, ys]))
    # Defensive: ensure positive semidefinite-ish
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.abs(vals))
    if np.any(np.isnan([width, height])):
        return
    e = Ellipse((x_m, y_m), width, height, angle=angle, facecolor=color, alpha=alpha, edgecolor=None)
    ax.add_patch(e)

# Add this before the update function definition
mean_history = {exp: {"x": [], "y": [], "rounds": []} for exp in experiments}

def update(frame_index):
    ax.cla()
    ax.set_xlim(0, xmax + xmargin)
    ax.set_ylim(0, ymax + ymargin)
    ax.set_xlabel("Resource Level")
    ax.set_ylabel("Population Credits")
    ax.set_title(f"Round {frame_index}")

    for exp in experiments:
        sub = df[(df["experiment"] == exp) & (df["round"] == frame_index)]
        if sub.empty:
            continue

        xs = sub["R_after_growth"].to_numpy()
        ys = sub["total_credits"].to_numpy()
        color = exp_to_color[exp]
        
        # Calculate current mean
        x_mean, y_mean = xs.mean(), ys.mean()

        # Update history for this experiment
        if frame_index not in mean_history[exp]["rounds"]:
            mean_history[exp]["x"].append(x_mean)
            mean_history[exp]["y"].append(y_mean)
            mean_history[exp]["rounds"].append(frame_index)

        # # Draw trail line connecting historical means
        # if len(mean_history[exp]["x"]) > 1:
        #     trail_x = mean_history[exp]["x"]
        #     trail_y = mean_history[exp]["y"]
        #     ax.plot(trail_x, trail_y, color=color, alpha=0.4, linewidth=2, 
        #             linestyle='-', zorder=2)
        # Replace the trail drawing section with this:
        if len(mean_history[exp]["x"]) > 1:
            trail_x = mean_history[exp]["x"]
            trail_y = mean_history[exp]["y"]
            
            # Draw segments with decreasing alpha
            for i in range(len(trail_x) - 1):
                alpha = 0.15 + 0.65 * (i / len(trail_x))  # Fade from 0.15 to 0.7
                ax.plot(trail_x[i:i+2], trail_y[i:i+2], 
                        color=color, alpha=alpha, linewidth=1.5, zorder=2)

        # Plot current data points
        ax.scatter(xs, ys, color=color, alpha=0.6, s=35)
        
        # Plot mean blob
        ax.scatter([x_mean], [y_mean],
                color=color, s=140, edgecolor="black", linewidth=0.8, zorder=4)

        # convex hull code
        pts = np.unique(np.vstack([xs, ys]).T, axis=0)
        if pts.shape[0] >= 3:
            try:
                hull = ConvexHull(pts, qhull_options="QJ")
                poly = pts[hull.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.06, zorder=0)
            except:
                draw_cov_ellipse(ax, xs, ys, color=color, alpha=0.06, n_std=1.5)
        elif pts.shape[0] == 2:
            ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.25, linewidth=6)

    return []


# repeat the last frame 5 times
frames = rounds + [rounds[-1]] * 5
ani = FuncAnimation(fig, update, frames=frames, interval=600, blit=False)
ani.save(OUT_PATH, fps=2, dpi=200)
print(f"[OK] Saved animation: {OUT_PATH}")

# 
