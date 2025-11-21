#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd


def analyze_round(round_log):
    ws_start = round_log["world_state_start"]
    ws_extr = round_log.get("world_state_post_extraction", {})
    ws_growth = round_log.get("world_state_post_growth", {})

    # --- Get alive lists ---
    alive_pre = round_log["agents_start"]["alive"]
    alive_post = round_log["agents_post_extraction"]["alive"]
    dead_post = round_log["agents_post_extraction"]["dead"]

    pre_names = {a["name"] for a in alive_pre}
    post_names = {a["name"] for a in alive_post}
    newly_dead = pre_names - post_names  # Agents who disappeared

    # === NEW METRICS ===
    # 1) Spawned (already computed)
    num_spawned = len(round_log.get("reproduction", []))

    # 2) Voted out (accusation‐based elimination)
    voted_out = []
    note = round_log.get("accusation_result_note", "")
    if isinstance(note, str) and "eliminated" in note:
        # example: "Member2 eliminated."
        parts = note.split()
        if len(parts) >= 2:
            voted_out.append(parts[0])  # "Member2"

    # 3) Broke (credit ≤ 0 elimination)
    broke = []
    for n in round_log.get("notes", []):
        if "ELIMINATION NOTICE" in n and "credit balance" in n:
            # find the name inside the note
            for alive_entry in alive_pre:
                if alive_entry["name"] in n:
                    broke.append(alive_entry["name"])

    # If agent died but not voted_out and not broke,
    # still treat as broke (fallback)
    for agent in newly_dead:
        if agent not in voted_out and agent not in broke:
            broke.append(agent)

    # ----------------------------------------------------------------------

    R_before = ws_start["R"]
    R_after_extraction = ws_extr.get("R", np.nan)
    R_after_growth = ws_growth.get("R", R_after_extraction)

    resource_delta_extraction = R_after_extraction - R_before
    resource_delta_growth = R_after_growth - R_after_extraction
    resource_sustainability_index = (
        R_after_growth / R_before if R_before else np.nan
    )

    Q = ws_start.get("quota", np.nan)
    fair_share = Q / len(alive_pre) if alive_pre else np.nan

    true_extracted = round_log.get("agents_extracted", {})
    if true_extracted:
        vals = list(true_extracted.values())
        extraction_discrepancy = max(vals) - min(vals)
    else:
        extraction_discrepancy = np.nan

    reported = round_log.get("agents_reported", {})
    total_true = sum(true_extracted.values())
    total_reported = sum(reported.values())

    deviation_true_vs_reported = {
        k: true_extracted[k] - reported.get(k, 0) for k in true_extracted
    }

    honesty_index = 1 - (
        sum(abs(v) for v in deviation_true_vs_reported.values()) / (total_true or 1)
    )

    penalties = round_log.get("extraction_penalties", {})
    total_penalty_cost = sum(penalties.values())
    audit_triggered = (
        total_penalty_cost > 0
        or any("audit" in n.lower() for n in round_log.get("notes", []))
    )

    credits_pre = [a["credits"] for a in alive_pre]
    credits_post = [a["credits"] for a in alive_post]
    credit_mean_pre = np.mean(credits_pre) if credits_pre else np.nan
    credit_mean_post = np.mean(credits_post) if credits_post else np.nan

    fairness_stdev = np.std(list(true_extracted.values())) if true_extracted else np.nan

    return {
        "agents_spawned": num_spawned,
        "agents_voted_out": len(voted_out),
        "agents_broke": len(broke),
        "agents_eliminated": len(voted_out) + len(broke),

        "num_alive_pre": len(alive_pre),
        "num_alive_post": len(alive_post),
        "voted_out_list": ",".join(voted_out),
        "broke_list": ",".join(broke),

        # everything else unchanged...
        "R_before": R_before,
        "R_after_extraction": R_after_extraction,
        "R_after_growth": R_after_growth,
        "resource_delta_extraction": resource_delta_extraction,
        "resource_delta_growth": resource_delta_growth,
        "resource_sustainability_index": resource_sustainability_index,
        "quota_total": Q,
        "fair_share": fair_share,
        "total_true_extraction": total_true,
        "total_reported_extraction": total_reported,
        "extraction_discrepancy": extraction_discrepancy,
        "honesty_index": honesty_index,
        "fairness_stdev": fairness_stdev,
        "audit_triggered": audit_triggered,
        "total_penalty_cost": total_penalty_cost,
        "credit_mean_pre": credit_mean_pre,
        "credit_mean_post": credit_mean_post,
        "credit_mean_delta": credit_mean_post - credit_mean_pre,
        "audit_threshold_T": ws_start.get("threshold"),
        "audit_probability_P": ws_start.get("audit_probability"),
        "penalty_multiplier_M": ws_start.get("penalty_multiplier"),
        "base_penalty_B": ws_start.get("base_penalty"),
        "alive_agents": len(alive_post),
    }


def analyze_run(run_index):
    """Loads a single run’s main_log.json and returns a DataFrame of per-round metrics."""
    run_dir = f"runs/run_{run_index}"
    log_path = os.path.join(run_dir, "main_log.json")

    if not os.path.exists(log_path):
        print(f"[WARN] Missing log for run {run_index}")
        return pd.DataFrame()

    with open(log_path, "r") as f:
        log_data = json.load(f)

    round_metrics = []
    for round_num, entry in enumerate(log_data, start=1):
        metrics = analyze_round(entry)
        metrics["run_id"] = run_index
        metrics["round"] = round_num
        round_metrics.append(metrics)

    df = pd.DataFrame(round_metrics)
    return df


def compute_run_level_aggregates(df):
    g = df.groupby("run_id")

    # First build the aggregate without the ratio
    summary = g.agg(
        num_rounds=("round", "max"),

        # population
        mean_alive=("alive_agents", "mean"),
        total_spawned=("agents_spawned", "sum"),
        total_eliminated=("agents_eliminated", "sum"),
        total_voted_out=("agents_voted_out", "sum"),
        total_broke=("agents_broke", "sum"),

        # resource & honesty
        mean_resource_ratio=("resource_sustainability_index", "mean"),
        mean_honesty=("honesty_index", "mean"),
        mean_fairness_stdev=("fairness_stdev", "mean"),

        # economy
        total_penalties=("total_penalty_cost", "sum"),
        mean_credit_delta=("credit_mean_delta", "mean"),

        # audit
        audit_prob_mean=("audit_probability_P", "mean"),

        # policy / environment
        mean_quota=("quota_total", "mean"),
        R_initial=("R_before", "first"),
        R_final=("R_after_growth", "last"),
    ).reset_index()

    # Now compute the ratio safely
    summary["final_resource_ratio"] = summary["R_final"] / summary["R_initial"]

    # Flag runs that lost more than half their resource
    summary["depleted"] = summary["R_final"] < 0.5 * summary["R_initial"]

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze CPR simulation runs.")
    parser.add_argument(
        "--runs", nargs="+", type=int, required=True, help="List of run indices to analyze"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output CSV filename"
    )
    args = parser.parse_args()

    all_dfs = []
    for run_index in args.runs:
        df = analyze_run(run_index)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No valid runs found.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"[OK] Wrote combined metrics for {len(all_dfs)} runs to {args.output}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"[OK] Wrote combined metrics for {len(all_dfs)} runs to {args.output}")

    # Create per-run summary
    run_summary = compute_run_level_aggregates(combined)
    run_summary.to_csv(args.output.replace(".csv", "_run_summary.csv"), index=False)
    print(f"[OK] Wrote run-level summary to {args.output.replace('.csv', '_run_summary.csv')}")


if __name__ == "__main__":
    main()
