import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull, QhullError

# EXPS = "regular"
# EXPS = "saboteur"
EXPS = "saboteur_with_snitches"
# EXPS = "bad_kimi_rollouts"


SUFF = "" if EXPS == "regular" else f"_{EXPS}"
PLOTDIR = Path("plots/quantitative" + SUFF)
DASHDIR = Path("plots/dashboards" + SUFF)
CSVDIR = Path(f"tables/{EXPS}")
RUNS = Path("runs")


PLOTDIR.mkdir(exist_ok=True, parents=True)
DASHDIR.mkdir(exist_ok=True, parents=True)
CSVDIR.mkdir(exist_ok=True, parents=True)

runs = json.loads(Path("completed_experiments.json").read_text())[EXPS]
# runs = {
#     (
#         "".join(
#             [c for c in k.lower().replace(" ", "_").replace(".", "").replace("-", "") if not c.isdigit()]
#         ),
#         k
#     ): v for k, v in runs.items()
# }
runs = {
    (
        k.lower().replace(" ", "_").replace(".", "").replace(",", "").replace("-", ""),
        k
    ): v for k, v in runs.items()
}
run_ids = []
for v in runs.values():
    run_ids += v
cmd = f"python analyse_quantitative.py --output {CSVDIR}/all_runs_metrics.csv --runs " + " ".join(list(map(str, run_ids)))
print(cmd)
os.system(cmd)

for i in run_ids:
    p = RUNS / f"run_{i}/run{i}.html"
    if not p.exists():
        os.system(f"python generate_conversation_html.py --run-path {p.parent}")
    else:
        print(f"Skipping as {p} already exists.")
for i in run_ids:
    p = RUNS / f"run_{i}/run{i}.png"
    if not p.exists():
        os.system(f"python render_dashboard.py --run-path {p.parent}/main_log.json --last-only")
    else:
        print(f"Skipping as {p} already exists.")

def stack_images():
    # Separator line thickness
    separator_thickness = 5

    for (filename_prefix, description), run_numbers in runs.items():
        # Load all images for this run
        images = []
        for run_num in run_numbers:
            img_path = f"runs/run_{run_num}/run{run_num}.png"
            try:
                img = Image.open(img_path)
                images.append(img)
                print(f"Loaded: {img_path}")
            except FileNotFoundError:
                print(f"Warning: {img_path} not found, skipping...")
        
        if not images:
            print(f"No images found for {filename_prefix}, skipping...")
            continue

        # Get dimensions (assuming all images have the same width)
        widths = [img.width for img in images]
        heights = [img.height for img in images]

        # Create a new image with the combined height plus separators
        total_width = max(widths)
        num_separators = len(images) - 1
        total_height = sum(heights) + (num_separators * separator_thickness)

        stacked_image = Image.new('RGB', (total_width, total_height), color='white')

        # Paste images vertically with black separators
        y_offset = 0
        for i, img in enumerate(images):
            stacked_image.paste(img, (0, y_offset))
            y_offset += img.height
            
            # Add black separator line (except after the last image)
            if i < len(images) - 1:
                draw = ImageDraw.Draw(stacked_image)
                draw.rectangle(
                    [(0, y_offset), (total_width, y_offset + separator_thickness)],
                    fill='black'
                )
                y_offset += separator_thickness

        # Save the stacked image
        output_path = DASHDIR / f"{filename_prefix}.png"
        stacked_image.save(output_path)
        print(f"Saved: {output_path}\n")

    print("All stacked images created successfully!")

stack_images()


name_to_model = {}
for (_, k), i_s in runs.items():
    for i in i_s:
        models = []
        for j in range(1, 5):
            model = []
            for p in Path(f"runs/run_{i}/Member{j}/").glob("*.json"):
                data = json.loads(p.read_text())
                model.append(data[0]["model"])
            assert all(model[0] == m for m in model)
            if len(model):
                models.append(model[0])
        name_to_model[(i, k)] = (models, len(list(Path(f"runs/run_{i}/states/").glob("*.json"))))


metrics_csv = CSVDIR / "all_runs_metrics.csv"
run_summary_csv = CSVDIR / "all_runs_metrics_run_summary.csv"
# -----------------------------------

# --- helper: covariance ellipse fallback ---
def draw_cov_ellipse(ax, xs, ys, color, alpha=0.07, n_std=1.5):
    if len(xs) < 2:
        return
    x_m, y_m = np.mean(xs), np.mean(ys)
    cov = np.cov(np.vstack([xs, ys]))
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
    e = Ellipse((x_m, y_m), width, height, angle=angle,
                facecolor=color, alpha=alpha, edgecolor=None)
    ax.add_patch(e)


df = pd.read_csv(metrics_csv)
run_summary = pd.read_csv(run_summary_csv)

# ---- NEW POPULATION METRICS ----

# total eliminated per round
df["agents_eliminated"] = df["agents_voted_out"] + df["agents_broke"]

# alive agents after reproduction:
# We reconstruct this because num_alive_post no longer exists
# If you DO have agents_post_reproduction info after extraction, modify accordingly
df["alive_agents"] = None

for run_id, sub in df.groupby("run_id"):
    alive = []  # running tally
    for i, row in sub.sort_values("round").iterrows():
        if pd.isna(row["agents_spawned"]): row["agents_spawned"] = 0
        if pd.isna(row["agents_eliminated"]): row["agents_eliminated"] = 0

        if len(alive) == 0:  # first round: assume starting alive = 4 or read from file
            # safer to detect from agents_start->alive if you stored it
            alive_count = getattr(row, "alive_start", 4)  # fallback to 4
        else:
            alive_count = alive[-1] + row["agents_spawned"] - row["agents_eliminated"]

        alive.append(alive_count)

    df.loc[sub.index, "alive_agents"] = alive



# Helper: Audit aggression score
def audit_aggression(P, M, T):
    T_norm = T / 100 if T else 0
    return P * M * (1 - T_norm)

df["audit_aggression"] = df.apply(
    lambda r: audit_aggression(r["audit_probability_P"], r["penalty_multiplier_M"], r["audit_threshold_T"]),
    axis=1
)


# --- Build run_id → experiment mapping ---
run_to_exp = {}
for (_, exp), run_list in runs.items():
    for r in run_list:
        run_to_exp[r] = exp

# attach to dataframe
df["experiment"] = df["run_id"].map(run_to_exp)
run_summary["experiment"] = run_summary["run_id"].map(run_to_exp)


# --- Assign a distinct color to each experiment ---
experiment_names = list([k for _, k in runs])
colors = cm.get_cmap('tab10', len(experiment_names))
exp_to_color = {exp: colors(i) for i, exp in enumerate(experiment_names)}


# --------- Reusable trajectory plotting function ---------
def plot_trajectory(metric, ylabel, filename):
    plt.figure(figsize=(8,5))
    sns.set_style("whitegrid")

    # Compute Round 0 per run if missing
    #   - R_before (resource)
    #   - total_credits_pre (credits)
    # We patch df inline so round=0 exists
    patched = []
    for run_id, sub in df.groupby("run_id"):
        if 0 not in sub["round"].values:
            # build round 0 row
            first = sub.iloc[0].copy()
            first["round"] = 0
            first["R_after_growth"] = first["R_before"]
            first["alive_agents"] = df[df["run_id"] == run_id]["alive_agents"].iloc[0]
            first["total_credits_post"] = first["credit_mean_pre"] * first["alive_agents"]
            first["honesty_index"] = 1.0
            first["agents_spawned"] = 0
            first["agents_eliminated"] = 0
            first["agents_voted_out"] = 0
            first["agents_broke"] = 0

            patched.append(first)
    if patched:
        df_extra = pd.DataFrame(patched)
        df_local = pd.concat([df, df_extra], ignore_index=True)
    else:
        df_local = df.copy()

    # Plot: thin lines per run
    for exp, exp_df in df_local.groupby("experiment"):
        color = exp_to_color[exp]
        for run_id, sub in exp_df.groupby("run_id"):
            plt.plot(sub["round"], sub[metric],
                     color=color,
                     alpha=0.25,
                     linewidth=1,
                     label=f"{exp} (run {run_id})")

        # Plot thick average
        avg = exp_df.groupby("round")[metric].mean()
        plt.plot(avg.index, avg.values, color=color, linewidth=3, label=exp, alpha=0.6)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over time")

    # Legend: only thick mean lines
    mean_handles = []
    mean_labels = []
    for exp in experiment_names:
        h, = plt.plot([], [], color=exp_to_color[exp], linewidth=3, alpha=0.6)
        mean_handles.append(h)
        mean_labels.append(exp)

    plt.legend(mean_handles, mean_labels, fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTDIR / f"{filename}.png", dpi=200)
    plt.close()


# Resource level over rounds
plot_trajectory("R_after_growth", "Resource level", "traj_resource")

# Compute total credits (post) as a per-round series
df["total_credits_post"] = df["alive_agents"] * df["credit_mean_post"]
plot_trajectory("total_credits_post", "Total agent credits", "traj_total_credits")

plot_trajectory("total_credits_post", "Total agent credits", "traj_total_credits")

# # Alive agent count
# plot_trajectory("alive_agents", "Alive agents", "traj_alive_agents")

# # Agent births (num_spawned per round)
# plot_trajectory("agents_spawned", "Agents spawned", "traj_births")
# plot_trajectory("agents_eliminated", "Agents eliminated", "traj_eliminated")
# plot_trajectory("agents_voted_out", "Agents voted out", "traj_voted_out")
# plot_trajectory("agents_broke", "Agents broke", "traj_broke")



# # Extraction discrepancy: max - min extraction per round
# df["extraction_discrepancy"] = df["total_true_extraction"] / df["num_alive_pre"]  # placeholder if no detail
# # Better: if you stored per-agent extraction detail in df, replace above with actual discrepancy
# # For now we treat average as proxy

# plot_trajectory("extraction_discrepancy", "Extraction discrepancy", "traj_extraction_discrepancy")

# Audit aggression
plot_trajectory("audit_aggression", "Audit aggression score", "traj_audit_aggression")

# Honesty index
plot_trajectory("honesty_index", "Honesty index", "traj_honesty")

print("[OK] Generated per-round trajectory plots.")




# ----------- (3) RESOURCE STABILITY vs COMMUNICATION COST -------------

# # Communication cost proxy: change in credits + penalties
# df["credit_change"] = df["credit_mean_delta"]
# final_df = df.groupby(["run_id","experiment"]).agg(
#     mean_comm_cost=("credit_change", lambda x: -x.sum()),  # negative -> cost
#     resource_change=("R_after_growth", lambda x: x.iloc[-1] - x.iloc[0]),
#     final_alive=("alive_agents", lambda x: x.iloc[-1])
# ).reset_index()

# plt.figure(figsize=(6,5))
# for exp, sub in final_df.groupby("experiment"):
#     plt.scatter(sub["mean_comm_cost"], sub["resource_change"], s=50, label=exp, alpha=0.8)

# plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
# plt.xlabel("Total communication cost (approx)")
# plt.ylabel("Resource change (final - initial)")
# plt.title("Communication cost vs resource stability")
# plt.legend()
# plt.tight_layout()
# plt.savefig(PLOTDIR / "comm_cost_vs_resource.png", dpi=200)
# plt.close()




# ----------- (2) COMBINED PER-RUN BAR CHARTS (3x2 GRID) -------------


# Collect all experiment names to build a shared palette
all_experiments = sorted(df["experiment"].unique())
palette = sns.color_palette("tab10", len(all_experiments))
exp_to_color = {exp: palette[i] for i, exp in enumerate(all_experiments)}

# Precomputed data for each plot
agg_df = df.groupby(["run_id","experiment"])["audit_aggression"].mean().reset_index()
agg_df = agg_df.groupby(["experiment"])["audit_aggression"].mean().reset_index()

fair_df = df.groupby(["run_id","experiment"])["extraction_discrepancy"].max().reset_index()
fair_df = fair_df.groupby(["experiment"])["extraction_discrepancy"].mean().reset_index()

elim_avg = df.groupby(["run_id","experiment"])["agents_eliminated"].mean().reset_index()
elim_avg = elim_avg.groupby(["experiment"])["agents_eliminated"].mean().reset_index()

voted_avg = df.groupby(["run_id","experiment"])["agents_voted_out"].mean().reset_index()
voted_avg = voted_avg.groupby(["experiment"])["agents_voted_out"].mean().reset_index()

broke_avg = df.groupby(["run_id","experiment"])["agents_broke"].mean().reset_index()
broke_avg = broke_avg.groupby(["experiment"])["agents_broke"].mean().reset_index()

born_avg = df.groupby(["run_id","experiment"])["agents_spawned"].mean().reset_index()
born_avg = born_avg.groupby(["experiment"])["agents_spawned"].mean().reset_index()

# ---------- Create one figure with subplots ----------
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.flatten()

plot_data = [
    (agg_df, "audit_aggression", "Audit Aggression by Experiment"),
    (fair_df, "extraction_discrepancy", "Extraction Unfairness by Experiment"),
    (elim_avg, "agents_eliminated", "Agent Elimination Rate"),
    (voted_avg, "agents_voted_out", "Voting Eliminations"),
    (broke_avg, "agents_broke", "Bankruptcies"),
    (born_avg, "agents_spawned", "Reproduction Rate")
]

for ax, (df_plot, col, title) in zip(axes, plot_data):
    sns.barplot(
        data=df_plot,
        y="experiment",
        x=col,
        ax=ax,
        palette=exp_to_color
    )
    ax.set_title(title)
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(PLOTDIR / "all_bars_combined.png", dpi=250)
plt.close()

print("[OK] Generated combined bar chart figure.")



# ---------- Compute summary per run ----------
df["credit_change"] = df["credit_mean_delta"]
final_df = df.groupby(["run_id","experiment"]).agg(
    mean_comm_cost=("credit_change", lambda x: -x.sum()),  # communication cost
    resource_change=("R_after_growth", lambda x: x.iloc[-1] - x.iloc[0]),
    final_alive=("alive_agents", lambda x: x.iloc[-1])
).reset_index()

fig, ax = plt.subplots(figsize=(8, 6))

for exp, sub in final_df.groupby("experiment"):
    xs = sub["mean_comm_cost"].to_numpy()
    ys = sub["resource_change"].to_numpy()
    color = exp_to_color[exp]

    # scatter points
    ax.scatter(xs, ys, s=40, color=color, alpha=0.6, label=exp)

    # mean blob
    x_mean, y_mean = xs.mean(), ys.mean()
    ax.scatter([x_mean], [y_mean], s=180, color=color,
               edgecolor="black", linewidth=0.8, zorder=3)

    # convex hull or ellipse
    pts = np.unique(np.vstack([xs, ys]).T, axis=0)
    if pts.shape[0] >= 3:
        try:
            hull = ConvexHull(pts, qhull_options="QJ")
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.06, zorder=0)
        except QhullError:
            draw_cov_ellipse(ax, xs, ys, color=color, alpha=0.06, n_std=1.5)
    elif pts.shape[0] == 2:
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.25, linewidth=6)

# axis labels and zero line
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Total communication cost (approx)")
ax.set_ylabel("Resource change (final - initial)")
ax.set_title("Communication cost vs resource stability")

# legend outside
handles = [plt.Line2D([], [], color=exp_to_color[exp], marker='o',
                       linestyle='', markersize=8) for exp in all_experiments]

ax.legend(handles, all_experiments, bbox_to_anchor=(1.02, 0.5),
          loc="center left", borderaxespad=0.)

plt.tight_layout()
plt.savefig(PLOTDIR / "comm_cost_vs_resource_hull_mean.png", dpi=250, bbox_inches="tight")
plt.close()




# --- per-run data ---
elim_per_run = df.groupby(["run_id","experiment"])["agents_eliminated"].mean().reset_index()
born_per_run = df.groupby(["run_id","experiment"])["agents_spawned"].mean().reset_index()

scatter_df = elim_per_run.merge(born_per_run, on=["run_id","experiment"], how="inner")

plt.figure(figsize=(8, 6))
ax = plt.gca()

for exp, sub in scatter_df.groupby("experiment"):
    xs = sub["agents_eliminated"].to_numpy()
    ys = sub["agents_spawned"].to_numpy()
    color = exp_to_color[exp]

    # Scatter points
    ax.scatter(xs, ys, s=40, color=color, alpha=0.6, label=exp)

    # Mean blob
    x_mean, y_mean = xs.mean(), ys.mean()
    ax.scatter([x_mean], [y_mean], s=180, color=color,
               edgecolor="black", linewidth=0.8, zorder=3)

    # Convex hull (fallback to ellipse)
    pts = np.unique(np.vstack([xs, ys]).T, axis=0)
    if pts.shape[0] >= 3:
        try:
            hull = ConvexHull(pts, qhull_options="QJ")
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.06, zorder=0)
        except QhullError:
            draw_cov_ellipse(ax, xs, ys, color=color, alpha=0.06, n_std=1.5)
    elif pts.shape[0] == 2:
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.25, linewidth=6)

ax.set_xlabel("Avg agents eliminated per round (per run)")
ax.set_ylabel("Avg agents spawned per round (per run)")
ax.set_title("Elimination vs Reproduction (per run)")

# legend outside right
handles = [plt.Line2D([], [], color=exp_to_color[exp], marker='o',
                       linestyle='', markersize=8) for exp in all_experiments]

plt.legend(handles, all_experiments, bbox_to_anchor=(1.02, 0.5),
           loc="center left", borderaxespad=0.)
plt.tight_layout()
plt.savefig(PLOTDIR / "elim_vs_born_scatter_hull_mean.png", dpi=250, bbox_inches="tight")
plt.close()

print("[OK] Generated scatter plot with hull + mean blobs.")
