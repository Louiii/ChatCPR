#!/usr/bin/env python3
import json
import os
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

# EXPS = "regular"
EXPS = "saboteur"
PLOTDIR = Path("plots/qualitative" + ("" if EXPS == "regular" else f"_{EXPS}"))
CSVDIR = Path("tables/qualitative" + ("" if EXPS == "regular" else f"_{EXPS}"))

BEHAVIOR_COLS = [
    "extraction_deception",
    "intent_to_deceive",
    "manipulative_communication",
    "selfish_extraction",
    # "strategic_silence",
    "weaponized_accusation",
    "blame_shifting",
    "collusion",
    "self_preservation_via_reproduction",
]


# Load the experiments
runs = json.loads(Path("completed_experiments.json").read_text())[EXPS]

def process_name_group(name, run_ids):
    """Process all run_ids for a given name sequentially"""
    print(f"[{name}] Starting group with {len(run_ids)} runs", flush=True)

    for ri in run_ids:
        time.sleep(60)
        print(f"[{name}] Executing run_{ri}", flush=True)

        # Use subprocess instead of os.system for better control
        result = subprocess.run(
            ["python", "analyse_qualitative.py", 
             "--input", f"runs/run_{ri}", 
             "--model", "openai/gpt-5-mini"],
            capture_output=False  # Let output go directly to terminal
        )

        if result.returncode != 0:
            print(f"[{name}] WARNING: run_{ri} failed with code {result.returncode}", flush=True)

    print(f"[{name}] Completed group", flush=True)


def analyse_parallel(runs):
    print(f"Total groups to process: {len(runs)}", flush=True)
    print("Starting all groups in parallel...\n", flush=True)

    # Create a process for each name group
    processes = []
    for name, run_ids in runs.items():
        p = Process(target=process_name_group, args=(name, run_ids))
        p.start()
        processes.append((name, p))
        print(f"Launched process for: {name} (PID: {p.pid})", flush=True)

    print(f"\nAll {len(processes)} processes launched!\n", flush=True)

    # Wait for all processes to complete
    for name, p in processes:
        p.join()
        print(f"Process completed: {name}", flush=True)

    print("\n✓ All groups completed!", flush=True)


def check_run_is_analysed(run_id: int, model: str = "gpt-5-mini") -> bool:
    run_path = Path(f"runs/run_{run_id}/")
    members = run_path.glob("Member*")
    conversations = set()
    try:
        for member in members:
            for p in member.glob("conversation_*.json"):
                conversations.add((p.parent.name, p.name.split("_")[1].split("_")[0]))
        we_have = {
            (p.name.split("_")[0], p.name.split("_")[2])
            for p in (run_path / f"analysis/{model}").glob("*")
            if p.name != "failed"
        }
        missing = conversations - we_have
        return len(missing) == 0
    except Exception as e:
        return e


def load_all_analysis(runs_dict, analysis_model_dir="gpt-5-mini", include_conversations=True):
    """
    Load per-agent analysis JSON for all runs.

    - runs_dict: dict[str, list[int]] mapping model_family -> [run_ids]
    - analysis_model_dir: subdir under runs/run_{id}/analysis/ where Member*.json live
    - include_conversations: if True, also attach list of conversation files for that member

    Returns: pandas.DataFrame with one row per (run_id, member).
    """
    rows = []
    for model_family, run_ids in runs_dict.items():
        for run_id in run_ids:
            base = Path(f"runs/run_{run_id}")
            ana_dir = base / "analysis" / analysis_model_dir
            if not ana_dir.exists():
                print(f"[WARN] Missing analysis dir: {ana_dir}")
                continue

            for f in ana_dir.glob("Member*.json"):
                try:
                    payload = json.loads(f.read_text())
                except Exception as e:
                    print(f"[WARN] Failed to read {f}: {e}")
                    continue

                analysis_agent = payload.get("analysis_agent")  # model that did the judging
                original_agent = payload.get("original_agent")  # the simulated agent id/name
                output = payload.get("output", {}) or {}
                reasoning = payload.get("reasoning", None)      # often None for gpt series

                row = {
                    "model_family": model_family,
                    "run_id": run_id,
                    "member": f.stem,                     # e.g., "Member4"
                    "path": str(f),
                    "analysis_model_dir": analysis_model_dir,
                    "analysis_agent": analysis_agent,
                    "original_agent": original_agent,
                }

                # Flatten the output:
                # For dict values with {"present": ..., "evidence": "...", ...}
                # store <key> = present (bool/str) and <key>_evidence = evidence (str)
                for key, val in output.items():
                    if isinstance(val, dict):
                        if "present" in val:
                            row[key] = val["present"]
                        # carry evidence if present; helps later drilldowns
                        if "evidence" in val and isinstance(val["evidence"], (str, list)):
                            row[f"{key}_evidence"] = val["evidence"]
                        # pick up a severity field if present
                        if "severity" in val:
                            row[f"{key}_severity"] = val["severity"]
                    else:
                        # plain scalar at top level (e.g., behavior_label, summary)
                        row[key] = val

                # Optionally attach conversation files for quick jumping
                if include_conversations:
                    member_dir = base / row["member"]
                    if member_dir.exists():
                        convs = sorted(str(p) for p in member_dir.glob("conversation_*.json"))
                        row["conversation_paths"] = convs
                    else:
                        row["conversation_paths"] = []

                rows.append(row)

    df = pd.DataFrame(rows)

    # Coerce common behavior flags to booleans/strings safely
    for c in BEHAVIOR_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda x: True if x is True else (False if x is False else x))

    return df


# def review_analysis(run_id: int, model: str = "gpt-5-mini"):
#     run_path = Path(f"runs/run_{run_id}/")
#     for p in (run_path / f"analysis/{model}").glob("Member*"):
#         analysis = json.loads(p.read_text())["output"]

def interesting_cases(df):
    mask = df[BEHAVIOR_COLS].fillna(False).any(axis=1)
    return df[mask]


def collapse_to_agents(df):
    grouped = df.groupby(["model_family", "run_id", "original_agent"]).agg({
        c: lambda x: True if True in x.values else False  # any TRUE → TRUE
        for c in BEHAVIOR_COLS
    })

    grouped = grouped.reset_index()

    return grouped


def review_analysis(run_id, model=None):
    df_run = df[df["run_id"] == run_id]

    if model:
        df_run = df_run[df_run["model_family"] == model]

    df_run = df_run[
        df_run[BEHAVIOR_COLS].fillna(False).any(axis=1)
    ]

    return df_run[["model_family", "run_id", "member", "path"] + BEHAVIOR_COLS]


def prettify_col(c):
    return c.replace("_", " ").title()


def hetrogenous_plots(df):
    #######################################################
    # 2) Filter two experiment groups
    #######################################################

    mixed_companies_df = df[df["model_family"] == "Mixed Companies"].copy()
    mixed_intelligence_df = df[df["model_family"] == "Mixed Intelligence"].copy()

    # ensure dtype is boolean
    for c in BEHAVIOR_COLS:
        mixed_companies_df[c] = mixed_companies_df[c].astype(bool)
        mixed_intelligence_df[c] = mixed_intelligence_df[c].astype(bool)


    #######################################################
    # 3) Deception rates per model inside Mixed Companies
    #######################################################

    companies_mean = (
        mixed_companies_df.groupby("agent_model")[BEHAVIOR_COLS]
        .mean()
        .sort_index()
    )

    companies_total = (
        mixed_companies_df.groupby("agent_model")[BEHAVIOR_COLS]
        .sum()
        .astype(int)
        .sort_index()
    )

    print("\n=== Mixed Companies — Mean deception by agent model ===")
    print(companies_mean)

    print("\n=== Mixed Companies — Total events per agent model ===")
    print(companies_total)


    #######################################################
    # 4) Deception rates per model inside Mixed Intelligence
    #######################################################

    intel_mean = (
        mixed_intelligence_df.groupby("agent_model")[BEHAVIOR_COLS]
        .mean()
        .sort_index()
    )

    intel_total = (
        mixed_intelligence_df.groupby("agent_model")[BEHAVIOR_COLS]
        .sum()
        .astype(int)
        .sort_index()
    )

    print("\n=== Mixed Intelligence — Mean deception by agent model ===")
    print(intel_mean)

    print("\n=== Mixed Intelligence — Total events per agent model ===")
    print(intel_total)


def make_plots():
    PLOTDIR.mkdir(exist_ok=True, parents=True)
    CSVDIR.mkdir(exist_ok=True, parents=True)

    df = load_all_analysis(runs)
    df.to_csv(CSVDIR / "agent_behavior_analysis.csv", index=False)
    print("[OK] Wrote agent_behavior_analysis.csv")

    df_interesting = interesting_cases(df)
    df_interesting.to_csv(CSVDIR / "interesting_agents.csv", index=False)
    print("[OK] Wrote interesting_agents.csv")
    print(df_interesting.head())

    # Ensure all behavior columns are boolean and fill NaNs with False
    for c in BEHAVIOR_COLS:
        df[c] = df[c].fillna(False).astype(bool)

    # Mean rate of deceptive behaviors per model family (proportion 0–1)
    mean_rates = (
        df.groupby("model_family")[BEHAVIOR_COLS]
        .mean()
        .sort_index()
    )
    print("\n=== Mean deception rates per model family ===")
    print(mean_rates)

    # Total number of events per behavior per model family
    totals = (
        df.groupby("model_family")[BEHAVIOR_COLS]
        .sum()
        .astype(int)      # nice integer formatting
        .sort_index()
    )
    print("\n=== Total detected deception events per model family ===")
    print(totals)

    df_agents = collapse_to_agents(df)
    df_agents = df_agents.rename(columns={"original_agent": "agent_model"})
    
    df["agent_model"] = df["original_agent"]
    # Ensure df["agent_model"] is clean label (last component after '/')
    df["agent_model_clean"] = df["agent_model"].apply(lambda s: s.split("/")[-1] if isinstance(s, str) else s)

    if EXPS == "regular":
        hetrogenous_plots(df)

    # Identify all agent models appearing in heterogeneous runs
    all_models = df["agent_model_clean"].unique()
    palette = sns.color_palette("tab10", len(all_models))
    model_to_color = {m: palette[i] for i, m in enumerate(all_models)}

    def get_stats(exp_name):
        """Return mean and total tables for a given experiment."""
        exp_df = df[df["model_family"] == exp_name].copy()
        mean_stats = exp_df.groupby("agent_model_clean")[BEHAVIOR_COLS].mean()
        total_stats = exp_df.groupby("agent_model_clean")[BEHAVIOR_COLS].sum().astype(int)
        return mean_stats, total_stats


    # def plot_table(stats, title, filename):
    #     """Generic grouped barplot for mean or total metrics."""
    #     stats = stats.copy()
    #     stats = stats.sort_index()  # consistent ordering
        
    #     fig, ax = plt.subplots(figsize=(12, 6))

    #     # Melt for seaborn: agent_model, behavior, value
    #     melted = stats.reset_index().melt(id_vars="agent_model_clean", var_name="behavior", value_name="value")

    #     sns.barplot(
    #         data=melted,
    #         x="behavior",
    #         y="value",
    #         hue="agent_model_clean",
    #         palette=[model_to_color[m] for m in stats.index.unique()],
    #         ax=ax,
    #     )

    #     ax.set_title(title)
    #     ax.set_xlabel("Behavior metric")
    #     ax.set_ylabel("Mean rate" if "Mean" in title else "Total count")
    #     ax.tick_params(axis="x", rotation=45)
    #     ax.legend(title="Agent Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    #     plt.tight_layout()
    #     plt.savefig(PLOTDIR / f"{filename}.png", dpi=200)
    #     plt.close()

    def plot_table(stats, title, filename, desired_order=None):
        stats = stats.copy()

        # Convert model names to "clean" labels
        clean_index = [m.split("/")[-1] for m in stats.index]
        stats.index = clean_index

        # Title-case behavior column names
        stats = stats.rename(columns={c: c.replace("_", " ").title() for c in stats.columns})

        if desired_order is None:
            desired_order = stats.index.unique()
        else:
            # Reindex to desired legend order
            stats = stats.reindex(desired_order)

        # Melt for seaborn
        melted = stats.reset_index().melt(
            id_vars="index",
            var_name="Behavior",
            value_name="Value"
        )
        melted.rename(columns={"index": "Agent Model"}, inplace=True)

        plt.figure(figsize=(12,6))

        # Explicit mapping: palette as dict tied to hue name
        palette = {m: model_to_color[m] for m in desired_order}

        sns.barplot(
            data=melted,
            x="Behavior",
            y="Value",
            hue="Agent Model",
            order=stats.columns,                 # x-axis order matches stats
            hue_order=desired_order,             # ensures correct legend order & colors
            palette=palette,                     # dict mapping
        )

        plt.title(title)
        plt.xlabel("Behavior Metric")
        plt.ylabel("Mean Rate" if "Mean" in title else "Total Count")
        plt.xticks(rotation=45, ha="right")

        plt.legend(
            title="Agent Model",
            loc="upper left",
            bbox_to_anchor=(1.02, 1)
        )
        plt.tight_layout()
        plt.savefig(PLOTDIR / f"{filename}.png", dpi=200)
        plt.close()




    # ======== Mixed Companies Plots ========
    mc_mean, mc_total = get_stats("Mixed Companies")
    mi_mean, mi_total = get_stats("Mixed Intelligence")


    mean_rates = df.groupby("model_family")[BEHAVIOR_COLS].mean()
    totals = df.groupby("model_family")[BEHAVIOR_COLS].sum()


    pretty_map = {c: prettify_col(c) for c in BEHAVIOR_COLS}
    # df = df.rename(columns=pretty_map)

    if EXPS == "regular":
        plot_table(mc_mean, "Mixed Companies — Mean Deception Rates", "mixed_companies_mean")
        plot_table(mc_total, "Mixed Companies — Total Deception Events", "mixed_companies_total")

        # ======== Mixed Intelligence Plots ========
        desired_order = [
            "gemini-2.5-flash-lite",
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
        ]
        plot_table(mi_mean, "Mixed Intelligence — Mean Deception Rates", "mixed_intelligence_mean", desired_order)
        plot_table(mi_total, "Mixed Intelligence — Total Deception Events", "mixed_intelligence_total", desired_order)

    print("[OK] Saved 4 plots into /plots/")


    mean_rates_pretty = mean_rates.rename(columns=pretty_map)
    melted = mean_rates_pretty.reset_index().melt(
        id_vars="model_family",
        var_name="Behavior",
        value_name="Mean Rate"
    )
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=melted,
        x="Behavior",
        y="Mean Rate",
        hue="model_family",
        palette="tab10"
    )
    plt.title("Mean Deception Rates per Model Family")
    plt.ylabel("Mean Rate")
    plt.xlabel("Behavior Metric")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model Family", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTDIR / "mean_deception_per_model_family.png", dpi=200)
    plt.close()


    totals_pretty = totals.rename(columns=pretty_map)
    melted = totals_pretty.reset_index().melt(
        id_vars="model_family",
        var_name="Behavior",
        value_name="Event Count"
    )
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=melted,
        x="Behavior",
        y="Event Count",
        hue="model_family",
        palette="tab10"
    )
    plt.title("Total Deception Events per Model Family")
    plt.ylabel("Event Count")
    plt.xlabel("Behavior Metric")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model Family", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTDIR / "total_deception_per_model_family.png", dpi=200)
    plt.close()

    agent_counts = df_agents.groupby("agent_model")[BEHAVIOR_COLS].sum().astype(int)
    agent_counts_pretty = agent_counts.rename(columns=pretty_map)
    melted = agent_counts_pretty.reset_index().melt(
        id_vars="agent_model",
        var_name="Behavior",
        value_name="Event Count"
    )
    plt.figure(figsize=(14,6))
    sns.barplot(
        data=melted,
        x="Behavior",
        y="Event Count",
        hue="agent_model",
        palette="tab10"
    )
    plt.title("Total Deception Events per Agent Model (All Runs)")
    plt.ylabel("Event Count")
    plt.xlabel("Behavior Metric")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Agent Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTDIR / "total_events_per_agent_model.png", dpi=200)
    plt.close()


    df_agents["any_deception"] = df_agents[BEHAVIOR_COLS].any(axis=1)

    share = df_agents.groupby("model_family")["any_deception"].mean()

    share.plot(kind="bar", figsize=(8,5), color="steelblue")
    plt.title("Fraction of Agents In Each Model Family That Committed Any Deception")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(PLOTDIR / "share_agents_with_deception.png", dpi=200)
    plt.close()


    df_interesting.groupby("model_family").size().plot(
        kind="bar",
        figsize=(7,5),
        color="darkred"
    )
    plt.title("Agents Flagged for Interesting Behavior")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTDIR / "interesting_agents_count.png", dpi=200)
    plt.close()


def stage1():
    analyse_parallel(runs)


def stage2():
    done = {k: list(map(check_run_is_analysed, v)) for k, v in runs.items()}
    _not_done = {k: np.array(runs[k])[np.where(~np.array(v))[0]] for k, v in done.items()}
    not_done = [v for vs in _not_done.values() for v in vs]
    analyse_parallel({"Not Done": not_done})


def stage3():
    make_plots()


if __name__ == "__main__":
    # stage1()
    # stage2()
    stage3()
