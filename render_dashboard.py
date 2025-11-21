from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import transforms
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

FRAMES_PER_TIMESTEP = 9  # (START ... WORLD_GROWTH) phases per timestep

class Phases(Enum):
    START = 0
    POST_CONVERSATION = 1
    POLICY_VOTE = 2
    POST_VOTE = 3
    EXTRACTION = 4
    WORLD_POST_EXTRACTION = 5
    AGENTS_POST_EXTRACTION = 6
    AGENTS_POST_REPRODUCTION = 7
    WORLD_GROWTH = 8


PolicyKey = str
AxesLike = Dict[str, plt.Axes]
SubplotParent = Union[plt.Figure, "matplotlib.gridspec.SubplotSpec"]


def parse_args():
    parser = argparse.ArgumentParser(description='Process run data')
    parser.add_argument(
        '--run-path',
        type=Path,
        help='Directory path (e.g. runs/run_1)'
    )
    parser.add_argument(
        '--bitrate',
        type=int,
        default=1000,
        help='Bitrate of the mp4 output (default: 3000)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=2,
        help='Frames per second of the output mp4 (default: 5)'
    )
    parser.add_argument(
        "--last-only",
        action="store_true",
        help="Whether to render a png image of the last frame"
    )
    return parser.parse_args()


def all_agents(timesteps: Sequence[dict]) -> List[str]:
    names = set()
    for ts in timesteps:
        for key in ["agents_start","agents_post_conversation","agents_post_extraction","agents_post_reproduction"]:
            if key in ts and "alive" in ts[key]:
                for a in ts[key]["alive"]:
                    names.add(a["name"])
        for key in ["agents_extracted","agents_reported","extraction_penalties"]:
            if key in ts:
                for n in ts[key].keys():
                    names.add(n)
    return sorted(list(names))


def max_credits_over_series(timesteps: Sequence[dict]) -> float:
    return max(
        (max(a["credits"] for a in v["alive"])
         for t in timesteps
         for k, v in t.items()
         if "agents" in k and "alive" in v),
        default=0,
    )


def policy_extents_from_timesteps(timesteps: Sequence[dict], policy_keys: Sequence[PolicyKey]) -> Dict[PolicyKey, List[float]]:
    ext = {k: [] for k in policy_keys}
    for ts in timesteps:
        ws0 = ts.get("world_state_start", {})
        wsv = ts.get("world_state_post_vote", {})
        for k in policy_keys:
            if k in ws0: ext[k].append(ws0[k])
            if k in wsv: ext[k].append(wsv[k])
        for vote in ts.get("policy_votes", {}).values():
            for k in policy_keys:
                if k in vote: ext[k].append(vote[k])
    return ext


def _credit_slot(state: DashboardState, agent: str, slot_index: int) -> float:
    """
    Safely fetch credits for `agent` at slot_index (0/1/2 for a given round).
    Returns np.nan if agent missing or slot is out of range.
    """
    arr = state.series.series_credits_time.get(agent)
    if arr is None:
        return np.nan
    if slot_index < 0 or slot_index >= arr.size:
        return np.nan
    return arr[slot_index]


@dataclass
class Lines:
    line_R: any
    line_Obs: any
    credit_lines_time: Dict[str, any]
    extracted_lines: Dict[str, any]
    reported_lines: Dict[str, any]
    quota_lines: Dict[str, any]          # NEW: dotted per-agent quota/agent
    line_avg_extracted: any              # NEW: bold avg extracted / alive
    line_avg_reported: any               # NEW: bold avg reported / alive
    policy_agg: Dict[PolicyKey, any]


@dataclass
class AxBundle:
    fig: plt.Figure
    ax_resource: plt.Axes
    ax_resource_time: plt.Axes
    ax_pie: plt.Axes
    ax_credits: plt.Axes
    ax_credits_time: plt.Axes
    ax_extr_rep: plt.Axes
    ax_policy: List[plt.Axes]  # [thresh, quota, audit, penmult, basepen]


@dataclass
class SeriesState:
    T: int
    agent_names: List[str]
    policy_keys: List[PolicyKey]
    series_R: np.ndarray
    series_Obs: np.ndarray
    series_extracted: Dict[str, np.ndarray]
    series_reported: Dict[str, np.ndarray]
    series_penalty: Dict[str, np.ndarray]
    series_credits_time: Dict[str, np.ndarray]
    xs_resource: np.ndarray
    xs_credits: np.ndarray
    policy_agg: Dict[PolicyKey, List[float]]

    # NEW:
    series_avg_extracted: np.ndarray            # shape (T,)
    series_avg_reported: np.ndarray             # shape (T,)
    series_quota_per_agent: Dict[str, np.ndarray]  # each shape (T,)


@dataclass
class DashboardState:
    axes: AxBundle
    lines: Lines
    fills: List[Rectangle]
    frames: List[Rectangle]
    avg_gap_poly: Optional[any]            # NEW: shaded area between avg extracted & avg reported
    series: SeriesState
    agent_colors: Dict[str, Tuple[float, float, float]]
    max_credits: float
    policy_keys: List[PolicyKey]
    mako_colors: Dict[str, Tuple[float, float, float]]

    # Previously added
    note_text: Optional[any] = None
    penalty_segments: List[any] = field(default_factory=list)
    reproduction_links: List[any] = field(default_factory=list)

    conv_notes: List[any] = field(default_factory=list)
    extr_notes: List[any] = field(default_factory=list)


def build_dashboard(
    parent: SubplotParent,
    *,
    T: int,
    agent_names: Sequence[str],
    agent_colors: Dict[str, Tuple[float, float, float]],
    max_credits: float,
    policy_extents: Dict[PolicyKey, Sequence[float]],
    policy_keys: Sequence[PolicyKey] = ("threshold","quota","audit_probability","penalty_multiplier","base_penalty"),
    y_max_extrrep: float,
) -> DashboardState:
    # Theme
    mako = sns.color_palette("mako", 9)
    COLOR_R_TOTAL, COLOR_R_OBS = mako[0], mako[2]
    COLOR_PENALTY = mako[8]
    theme = dict(R_TOTAL=COLOR_R_TOTAL, R_OBS=COLOR_R_OBS, PENALTY=COLOR_PENALTY)

    # --- Build axes layout inside the parent region ---
    if isinstance(parent, plt.Figure):
        fig = parent
        gs = fig.add_gridspec(
            nrows=2, ncols=4,
            height_ratios=[1.0, 1.0],
            width_ratios=[1, 1, 1, 0.4],
            hspace=0.3, wspace=0.2,
        )
    else:
        gspec = parent.get_gridspec()
        fig = gspec.figure
        gs = parent.subgridspec(
            nrows=2, ncols=4,
            height_ratios=[1.0, 1.0],
            width_ratios=[1, 1, 1, 0.4],
            hspace=0.3, wspace=0.2,
        )

    sub_gs = gs[0, :3].subgridspec(1, 3, width_ratios=[0.8,3,2])
    ax_resource, ax_extr_rep, ax_pie = [fig.add_subplot(sub_gs[0, i]) for i in range(3)]

    sub_gs = gs[1, :2].subgridspec(1, 2, hspace=0., width_ratios=[1,5])
    ax_credits, ax_credits_time = [fig.add_subplot(sub_gs[0, i]) for i in range(2)]

    sub_gs = gs[1, 2].subgridspec(2, 1, hspace=0., height_ratios=[1, 1])
    ax_resource_time = fig.add_subplot(sub_gs[:2, 0])

    sub_gs = gs[:, 3].subgridspec(5, 1, hspace=0., height_ratios=[1,1,1,1,1])
    ax_policy = [fig.add_subplot(sub_gs[i, 0]) for i in range(5)]
    for ax in ax_policy:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
    ax_thresh, ax_quota, ax_audit, ax_penmult, ax_basepen = ax_policy

    # --- Configure axes & artists that don't depend on data yet ---
    # Resource (battery)
    ax_resource.set_xlim(-0.5, 1.5)
    ax_resource.set_ylim(0, 100)
    ax_resource.set_xticks([0,1])
    ax_resource.set_xticklabels(["Total", "Observed"])
    for spine in ax_resource.spines.values(): spine.set_visible(False)
    ax_resource.tick_params(axis='both', length=0)
    ax_resource.set_yticks([])
    ax_resource.set_title("Resource Levels", fontsize=11)

    frames = []
    for x in [0, 1]:
        frame = Rectangle((x-0.15, 0), 0.5, 100, fill=False, linewidth=1.8, edgecolor='black')
        ax_resource.add_patch(frame)
        frames.append(frame)

    battery_fills = [
        Rectangle((0-0.15, 0), 0.5, 0, linewidth=0, alpha=0.75, facecolor=COLOR_R_TOTAL),
        Rectangle((1-0.15, 0), 0.5, 0, linewidth=0, alpha=0.75, facecolor=COLOR_R_OBS),
    ]
    ax_resource.add_patch(battery_fills[0]); ax_resource.add_patch(battery_fills[1])

    # Resource over time
    (line_R,)  = ax_resource_time.plot([], [], marker="o", linestyle="-",
                                       label="Total Resource", color=COLOR_R_TOTAL)
    (line_Obs,) = ax_resource_time.plot([], [], marker="s", linestyle="-",
                                        label="Observed", color=COLOR_R_OBS)
    ax_resource_time.set_ylim(0, 100)
    ax_resource_time.set_xlim(1, T + 0.5)
    ax_resource_time.set_ylabel("Units")
    ax_resource_time.yaxis.tick_right()
    ax_resource_time.yaxis.set_label_position("right")
    ax_resource_time.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax_resource_time.legend(loc="best", frameon=True)

    # Credits over time (fractional x-positions)
    ax_credits_time.set_ylabel("Agent Credits")
    ax_credits_time.set_xlim(1, max(1, T) + 1)  # allow drawing to r+1 for reproduction links
    ax_credits_time.set_ylim(0, max_credits * 1.1 if max_credits > 0 else 1)
    ax_credits_time.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax_credits_time.yaxis.tick_right()
    ax_credits_time.yaxis.set_label_position("right")
    ax_credits_time.spines["left"].set_visible(False)
    ax_credits_time.spines["right"].set_visible(True)

    # Extracted vs reported (per-agent + totals)
    ax_extr_rep.set_title("Resource Extracted / Reported")
    # ax_extr_rep.set_xlim(1 * 0.95, max(1, T) * 1.05)
    ax_extr_rep.set_xlim(1, max(1, T))
    # ax_extr_rep.set_xlabel("Timestep")
    ax_extr_rep.set_ylabel("Units")
    ax_extr_rep.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax_extr_rep.yaxis.tick_right()
    ax_extr_rep.yaxis.set_label_position("right")
    ax_extr_rep.set_ylim(0.0, y_max_extrrep if y_max_extrrep > 0 else 1)
    ax_extr_rep.set_xlabel("")
    ax_extr_rep.get_xaxis().set_visible(False)
    # ax_extr_rep.margins(x=0.02, y=0.04)


    # Policy stacks
    ax_thresh.set_title("Policy Params")
    policy_axes = [ax_thresh, ax_quota, ax_audit, ax_penmult, ax_basepen]
    policy_titles = ["Threshold", "Quota", "Audit Probability", "Penalty Multiplier", "Base Penalty"]
    for i, ax in enumerate(policy_axes):
        ax.set_ylabel(policy_titles[i], fontsize=7, rotation=80)
        ax.set_xlim(1, max(1, T))
        k = policy_keys[i]
        vals = list(policy_extents.get(k, []))
        if vals:
            lo, hi = min(vals), max(vals)
            if lo == hi:
                lo, hi = (lo * 0.9, hi * 1.1) if hi != 0 else (0, 1)
            ax.set_ylim(lo * 0.9, hi * 1.1)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.tick_params(labelsize=8)
        if i < 4:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
        # IMPORTANT: no legend — we’ll show one aggregated line and per-agent scatters

    # Create one aggregated line per policy (no legend)
    policy_lines_agg = {}
    for k, ax in zip(policy_keys, policy_axes):
        # aggregated line lives at integer x = r (set in POST_VOTE)
        (l,) = ax.plot([], [], marker="o", linestyle="-", linewidth=1.5, color="black")
        policy_lines_agg[k] = l

    # Per-agent lines
    credit_lines_time = {}
    extracted_lines = {}
    reported_lines = {}
    for agent in agent_names:
        c = agent_colors[agent]
        (lc,) = ax_credits_time.plot([], [], marker="o", linestyle="-", label=agent, color=c)
        credit_lines_time[agent] = lc
        (le,) = ax_extr_rep.plot([], [], marker=".", linestyle="-", label=f"{agent} extracted", color=c, alpha=0.9)
        extracted_lines[agent] = le
        (lr,) = ax_extr_rep.plot([], [], marker=".", linestyle="--", label=f"{agent} reported", color=c, alpha=0.9)
        reported_lines[agent]  = lr

    # dotted quota/agent per agent (same color)
    quota_lines = {}
    for agent in agent_names:
        c = agent_colors[agent]
        (lq,) = ax_extr_rep.plot([], [], linestyle=":", marker=None, label=f"{agent} quota/agent", color=c, alpha=0.9)
        quota_lines[agent] = lq

    # bold overlays for per-alive averages
    (line_avg_extracted,) = ax_extr_rep.plot([], [], linestyle="-",  linewidth=2.0, color="black", alpha=0.8)
    (line_avg_reported,)  = ax_extr_rep.plot([], [], linestyle="--", linewidth=2.0, color="grey",  alpha=0.9)

    # X positions
    xs_resource = np.repeat(np.arange(1, T+1), 2).astype(float)
    xs_resource[1::2] += 0.5
    xs_credits = np.concatenate([
        np.array([r, r + 1/3, r + 2/3], dtype=float) for r in range(1, T+1)
    ])

    legend_handles = [
        Line2D([0], [0], linestyle='-',  color='black',  label='Extracted (per agent)'),
        Line2D([0], [0], linestyle='--', color='black',  label='Reported (per agent)'),
        Line2D([0], [0], linestyle=':',  color='black',  label='Quota per agent'),
        Line2D([0], [0], linestyle='-',  color='black',  linewidth=2.0, alpha=0.8, label='Avg extracted / alive'),
        Line2D([0], [0], linestyle='--', color='grey',   linewidth=2.0, alpha=0.9, label='Avg reported / alive'),
    ]
    ax_extr_rep.legend(handles=legend_handles, loc="best", frameon=True, fontsize=8)

    # Series (NaNs until filled)
    series_R   = np.full(2*T, np.nan)
    series_Obs = np.full(2*T, np.nan)
    series_extracted = {a: np.full(T, np.nan) for a in agent_names}
    series_reported  = {a: np.full(T, np.nan) for a in agent_names}
    series_penalty   = {a: np.full(T, np.nan) for a in agent_names}
    series_credits_time = {a: np.full(3*T, np.nan) for a in agent_names}
    series_avg_extracted = np.full(T, np.nan)
    series_avg_reported  = np.full(T, np.nan)
    series_quota_per_agent = {a: np.full(T, np.nan) for a in agent_names}


    fig.suptitle(f"Round 1 / {T} — {Phases.START.name}", fontsize=14)
    axes = AxBundle(
        fig=fig,
        ax_resource=ax_resource,
        ax_resource_time=ax_resource_time,
        ax_pie=ax_pie,
        ax_credits=ax_credits,
        ax_credits_time=ax_credits_time,
        ax_extr_rep=ax_extr_rep,
        ax_policy=policy_axes,
    )
    # build Lines
    lines = Lines(
        line_R=line_R,
        line_Obs=line_Obs,
        credit_lines_time=credit_lines_time,
        extracted_lines=extracted_lines,
        reported_lines=reported_lines,
        quota_lines=quota_lines,
        line_avg_extracted=line_avg_extracted,
        line_avg_reported=line_avg_reported,
        policy_agg=policy_lines_agg,
    )
    pol_agg   = {k: [np.nan]*T for k in policy_keys}
    series = SeriesState(
        T=T,
        agent_names=list(agent_names),
        policy_keys=list(policy_keys),
        series_R=series_R,
        series_Obs=series_Obs,
        series_extracted=series_extracted,
        series_reported=series_reported,
        series_penalty=series_penalty,
        series_credits_time=series_credits_time,
        xs_resource=xs_resource,
        xs_credits=xs_credits,
        policy_agg=pol_agg,
        series_avg_extracted=series_avg_extracted,
        series_avg_reported=series_avg_reported,
        series_quota_per_agent=series_quota_per_agent,
    )

    state = DashboardState(
        axes=axes,
        lines=lines,
        fills=battery_fills,
        frames=frames,
        avg_gap_poly=None,
        series=series,
        agent_colors=agent_colors,
        max_credits=max_credits,
        policy_keys=list(policy_keys),
        mako_colors=theme,
        conv_notes=[],
        extr_notes=[],
    )
    return state


def _set_note(state: DashboardState, text: str) -> None:
    if state.note_text is not None:
        state.note_text.set_text(text or "")

def _short_segment(ax: plt.Axes, x1, y1, x2, y2, *, color="red", lw=2.0, ls="-"):
    (seg,) = ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, zorder=4)
    return seg


def _draw_credits_bar_horizontal(state: DashboardState, agent_list: List[dict]) -> None:
    ax = state.axes.ax_credits
    ax.cla()
    names = [a["name"] for a in agent_list]
    vals  = [a["credits"] for a in agent_list]
    order = np.argsort(vals)
    names_sorted = [names[i] for i in order]
    vals_sorted  = [vals[i]  for i in order]

    y = np.arange(len(names_sorted))
    bars = ax.barh(y, vals_sorted, edgecolor="black", linewidth=1.0)
    for rect, name in zip(bars, names_sorted):
        rect.set_color(state.agent_colors.get(name, 'C0'))

    ax.set_axis_off()
    x_max = max(state.max_credits, max(vals_sorted) if len(vals_sorted) else 0)
    ax.set_xlim(x_max * 1.05 if x_max > 0 else 1, 0)


def _update_battery(state: DashboardState, R: float, err: Optional[float]) -> None:
    state.fills[0].set_height(max(0, min(100, R)))
    obs = R + (err if err is not None else 0.0)
    state.fills[1].set_height(max(0, min(100, obs)))


def _update_resource_time(state: DashboardState) -> None:
    xs = state.series.xs_resource
    state.lines.line_R.set_data(xs, state.series.series_R)
    state.lines.line_Obs.set_data(xs, state.series.series_Obs)
    ax = state.axes.ax_resource_time
    ax.relim(); ax.autoscale_view()


def _update_credits_time(state: DashboardState) -> None:
    xs = state.series.xs_credits
    for agent in state.series.agent_names:
        state.lines.credit_lines_time[agent].set_data(xs, state.series.series_credits_time[agent])
    ax = state.axes.ax_credits_time
    ax.relim(); ax.autoscale_view()


def _update_extr_rep(state: DashboardState) -> None:
    xs = np.arange(1, state.series.T + 1)

    # per-agent extracted / reported
    for agent in state.series.agent_names:
        state.lines.extracted_lines[agent].set_data(xs, state.series.series_extracted[agent])
        state.lines.reported_lines[agent].set_data(xs, state.series.series_reported[agent])

    # per-agent quota/agent
    for agent in state.series.agent_names:
        state.lines.quota_lines[agent].set_data(xs, state.series.series_quota_per_agent[agent])

    # averages
    state.lines.line_avg_extracted.set_data(xs, state.series.series_avg_extracted)
    state.lines.line_avg_reported.set_data(xs, state.series.series_avg_reported)

    # shaded region between averages
    ax = state.axes.ax_extr_rep
    if state.avg_gap_poly is not None:
        try:
            state.avg_gap_poly.remove()
        except Exception:
            pass

    y1 = state.series.series_avg_extracted
    y2 = state.series.series_avg_reported
    state.avg_gap_poly = ax.fill_between(xs, y1, y2, alpha=0.12, step='mid')

    ax.relim(); ax.autoscale_view()
    ax.margins(x=0.03, y=0.08)


def _update_pie(state: DashboardState, ts: dict) -> None:
    ax = state.axes.ax_pie
    ax.cla()
    data = ts.get("agents_extracted", {})
    if not data:
        ax.text(0.5, 0.5, "No extraction yet", ha='center', va='center')
        return
    labels, sizes, colors = [], [], []
    for agent in state.series.agent_names:
        v = data.get(agent, 0)
        if v and v > 0:
            labels.append(agent)
            sizes.append(v)
            colors.append(state.agent_colors[agent])
    if not sizes:
        ax.text(0.5, 0.5, "No extraction yet", ha='center', va='center')
        return
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f', startangle=90,
        colors=colors, pctdistance=0.8
    )
    for i, txt in enumerate(texts):
        txt.set_color(state.agent_colors[labels[i]])
    ax.set_title("Extraction Share")


def prime(state: DashboardState, t0: int, timesteps: Sequence[dict]) -> None:
    """Fill the dashboard with initial data so all artists are valid."""
    ts0 = timesteps[t0]
    w0 = ts0["world_state_start"]
    R0 = w0["R"]; err0 = w0.get("error", 0.0)
    _update_battery(state, R0, err0)
    _draw_credits_bar_horizontal(state, ts0["agents_start"]["alive"])

    # Ensure policy aggregated lines have initial NaNs bound
    xs = range(1, state.series.T+1)
    for k in state.policy_keys:
        state.lines.policy_agg[k].set_data(xs, state.series.policy_agg[k])

    # Credits: initial START slot (index 0 in 3*t grid)
    start_map = {a["name"]: a["credits"] for a in ts0.get("agents_start", {}).get("alive", [])}
    for agent in state.series.agent_names:
        state.series.series_credits_time[agent][0] = start_map.get(agent, np.nan)

    _update_resource_time(state)
    _update_credits_time(state)
    _update_pie(state, ts0)
    _set_note(state, "")   # empty initial note


def _add_note(
    state: DashboardState,
    *,
    x_data: float,
    header: Optional[str] = None,
    penalty_lines: Optional[List[str]] = None,
    rotate_header_deg: float = 0.0,
    y_header: float = 0.95,
    y_pen: float = 0.90,
):
    """
    Draws small text patches near the *top* of the credits-time axis at x=x_data.
    Uses a blended transform: x in data coordinates, y as an axes fraction (stays near top
    regardless of y-limits). Returns list of created Text artists.
    """
    ax = state.axes.ax_credits_time
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    artists = []
    if header:
        t = ax.text(
            x_data, y_header, header,
            transform=trans, ha="center", va="top",
            fontsize=8, rotation=rotate_header_deg,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5, alpha=0.9)
        )
        artists.append(t)

    if penalty_lines:
        txt = "\n".join(penalty_lines)
        t2 = ax.text(
            x_data, y_pen, txt,
            transform=trans, ha="center", va="top",
            fontsize=7, color="red"
        )
        artists.append(t2)

    return artists


def make_update_fn(
    state: DashboardState,
    timesteps: Sequence[dict],
    time_notes: bool = True,
) -> Tuple[Callable[[int, Phases], None], Callable[[int], List]]:
    def credits_slot_index(t: int, slot: int) -> int:
        return 3*t + slot  # 0=START, 1=POST_CONV, 2=POST_EXTR

    def alive_map(ts: dict, key: str) -> Dict[str, float]:
        return {a["name"]: a["credits"] for a in ts.get(key, {}).get("alive", [])}

    def _short_segment(ax: plt.Axes, x1, y1, x2, y2, *, color="red", lw=2.0, ls="-"):
        (seg,) = ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, zorder=4)
        return seg

    def update_by_phase(t: int, phase: Phases) -> None:
        ts = timesteps[t]
        r = t + 1

        if phase is Phases.START:
            R = ts["world_state_start"]["R"]
            err = ts["world_state_start"].get("error", 0.0)
            _update_battery(state, R, err)

            i = 2*t
            state.series.series_R[i]   = R
            state.series.series_Obs[i] = R + (err if err is not None else 0.0)
            _update_resource_time(state)
            _draw_credits_bar_horizontal(state, ts["agents_start"]["alive"])

            # Credits @ r
            start = alive_map(ts, "agents_start")
            k = credits_slot_index(t, 0)
            for agent in state.series.agent_names:
                state.series.series_credits_time[agent][k] = start.get(agent, np.nan)
            _update_credits_time(state)

        elif phase is Phases.POST_CONVERSATION:
            if "agents_post_conversation" in ts:
                _draw_credits_bar_horizontal(state, ts["agents_post_conversation"]["alive"])
                # Credits @ r+1/3
                post_conv = alive_map(ts, "agents_post_conversation")
                k = credits_slot_index(t, 1)
                for agent in state.series.agent_names:
                    state.series.series_credits_time[agent][k] = post_conv.get(agent, np.nan)
                _update_credits_time(state)

                if time_notes:
                    # Sticky note: "post conversation" rotated to fit
                    x_note = r + 0.1
                    artists = _add_note(state, x_data=x_note, header="post\nconv", rotate_header_deg=0.0, y_header=0.9)
                    state.conv_notes.extend(artists)

        elif phase is Phases.POLICY_VOTE:
            votes = ts.get("policy_votes", {})
            x = (t + 1) + 0.5  # r + 0.5
            ax_thresh, ax_quota, ax_audit, ax_penmult, ax_basepen = state.axes.ax_policy
            for agent, vote in votes.items():
                c = state.agent_colors.get(agent, None)
                for kpol, ax in zip(
                    state.policy_keys,
                    [ax_thresh, ax_quota, ax_audit, ax_penmult, ax_basepen]
                ):
                    if kpol in vote:
                        ax.scatter([x], [vote[kpol]], color=c, marker='.', zorder=3, s=18)

        # elif phase is Phases.POST_VOTE:

        elif phase is Phases.EXTRACTION:
            post = ts.get("world_state_post_vote", {})
            # aggregated policy line at integer r points (keep your existing code)
            for kpol in state.policy_keys:
                state.series.policy_agg[kpol][t] = post.get(kpol, np.nan)
            x_now = np.arange(1, t + 2, dtype=float)  # 1..r
            for kpol in state.policy_keys:
                y_now = np.asarray(state.series.policy_agg[kpol][:t+1], dtype=float)
                state.lines.policy_agg[kpol].set_data(x_now, y_now)

            # --- write per-agent extracted & reported for this timestep ---
            for agent in state.series.agent_names:
                state.series.series_extracted[agent][t] = ts.get("agents_extracted", {}).get(agent, np.nan)
                state.series.series_reported[agent][t]  = ts.get("agents_reported",  {}).get(agent, np.nan)

            # --- compute averages & per-agent quota/agent ---
            alive_list = ts.get("agents_post_extraction", {}).get("alive", []) or \
                        ts.get("agents_start", {}).get("alive", [])
            alive_names = {a["name"] for a in alive_list}
            n_alive = len(alive_names) if alive_names else 0

            if n_alive > 0:
                tot_ex = np.nansum([ts.get("agents_extracted", {}).get(a, np.nan) for a in alive_names])
                tot_rp = np.nansum([ts.get("agents_reported",  {}).get(a, np.nan) for a in alive_names])
                state.series.series_avg_extracted[t] = tot_ex / n_alive
                state.series.series_avg_reported[t]  = tot_rp / n_alive
            else:
                state.series.series_avg_extracted[t] = np.nan
                state.series.series_avg_reported[t]  = np.nan

            quota = ts.get("world_state_post_vote", {}).get("quota", np.nan)
            q_per = (quota / n_alive) if (n_alive and not np.isnan(quota)) else np.nan
            for agent in state.series.agent_names:
                state.series.series_quota_per_agent[agent][t] = q_per if agent in alive_names else np.nan

            # --- now draw the subplot (per-agent lines, averages, shaded gap) ---
            _update_extr_rep(state)

            # keep your pie update and credits time updates
            _update_pie(state, ts)

            # Credits @ r+2/3 and notes (keep your existing code below)
            post_ex = alive_map(ts, "agents_post_extraction")
            kslot_ex = credits_slot_index(t, 2)
            for agent in state.series.agent_names:
                state.series.series_credits_time[agent][kslot_ex] = post_ex.get(agent, np.nan)
            _update_credits_time(state)

            x_note = r + 2/3
            penalties = ts.get("extraction_penalties", {}) or {}
            penalty_lines = [f"{name}: -{pen:.2f}" for name, pen in penalties.items() if pen and pen > 0]
            artists = _add_note(
                state,
                x_data=x_note,
                header="post extraction" if time_notes else "",
                penalty_lines=penalty_lines if penalty_lines else None,
                rotate_header_deg=0.0
            )
            state.extr_notes.extend(artists)


        elif phase is Phases.WORLD_POST_EXTRACTION:
            post_ex = ts.get("world_state_post_extraction", {})
            R = post_ex.get("R", np.nan)
            err = post_ex.get("error", 0.0)
            if not np.isnan(R):
                _update_battery(state, R, err)
                i = 2*t + 1
                state.series.series_R[i]   = R
                state.series.series_Obs[i] = R + (err if err is not None else 0.0)
                _update_resource_time(state)
            for agent in state.series.agent_names:
                state.series.series_penalty[agent][t] = ts.get("extraction_penalties", {}).get(agent, np.nan)
            # _update_penalties(state)

        elif phase is Phases.AGENTS_POST_EXTRACTION:
            if "agents_post_extraction" in ts:
                _draw_credits_bar_horizontal(state, ts["agents_post_extraction"]["alive"])

        elif phase is Phases.AGENTS_POST_REPRODUCTION:
            if "agents_post_reproduction" in ts:
                _draw_credits_bar_horizontal(state, ts["agents_post_reproduction"]["alive"])

            # Reproduction links from (r+2/3, parent) -> (r+1, child)
            repro_list = ts.get("reproduction", []) or []
            if repro_list:
                # child credits read from next round's agents_start (or agents_post_reproduction fallback)
                next_map = {}
                if t + 1 < state.series.T:
                    next_map = {a["name"]: a["credits"] for a in timesteps[t+1].get("agents_start", {}).get("alive", [])}
                if not next_map:
                    next_map = {a["name"]: a["credits"] for a in ts.get("agents_post_reproduction", {}).get("alive", [])}

                ax = state.axes.ax_credits_time
                for entry in repro_list:
                    # expected: "MemberX spawned MemberY"
                    parts = entry.split()
                    if len(parts) < 3:
                        continue
                    parent = parts[0]
                    child  = parts[-1]

                    # parent at slot 2 (r+2/3)
                    y_parent = _credit_slot(state, parent, 3*t + 2)
                    y_child  = next_map.get(child, np.nan)

                    if np.isnan(y_parent) or np.isnan(y_child):
                        continue

                    x_parent = (t + 1) + 2/3
                    x_child  = (t + 1) + 1
                    color = state.agent_colors.get(parent, "C0")
                    (seg,) = ax.plot([x_parent, x_child], [y_parent, y_child],
                                    linestyle="--", linewidth=1.5, color=color, zorder=4)
                    state.reproduction_links.append(seg)


        elif phase is Phases.WORLD_GROWTH:
            post_gr = ts.get("world_state_post_growth", {})
            R = post_gr.get("R", np.nan)
            err = post_gr.get("error", 0.0)
            if not np.isnan(R):
                _update_battery(state, R, err)

        state.axes.fig.suptitle(f"Round {r} / {state.series.T} — {phase.name}", fontsize=14)

    def update_frame(frame: int):
        t = frame // FRAMES_PER_TIMESTEP
        phase = Phases(frame % FRAMES_PER_TIMESTEP)
        update_by_phase(t, phase)
        return []
    return update_by_phase, update_frame


def setup_from_timesteps(
    timesteps: Sequence[dict],
    parent: SubplotParent,
    *,
    palette_name: str = "rocket",
    time_notes: bool = True,
) -> Tuple[DashboardState, Callable[[int, Phases], None], Callable[[int], List]]:
    """
    Computes derived inputs (agents, colors, extents), builds the dashboard in `parent`,
    primes it, and returns (state, update_by_phase, update_frame).
    """
    T = len(timesteps)
    agents = all_agents(timesteps)
    palette = sns.color_palette(palette_name, len(agents) or 1)
    agent_colors = {name: palette[i % len(palette)] for i, name in enumerate(agents)}
    mx_cred = max_credits_over_series(timesteps)
    policy_keys = ["threshold","quota","audit_probability","penalty_multiplier","base_penalty"]
    extents = policy_extents_from_timesteps(timesteps, policy_keys)
    def _max(d): return max(d.values()) if d else 0
    y_max_extrrep = max(
        max((_max(ts.get("agents_extracted", {})) for ts in timesteps), default=0),
        max((_max(ts.get("agents_reported",  {})) for ts in timesteps), default=0),
    )

    # Accept Figure or SubplotSpec
    state = build_dashboard(
        parent,
        T=T,
        agent_names=agents,
        agent_colors=agent_colors,
        max_credits=mx_cred,
        policy_extents=extents,
        policy_keys=policy_keys,
        y_max_extrrep=y_max_extrrep * 1.05,
    )
    prime(state, 0, timesteps)
    update_by_phase, update_frame = make_update_fn(state, timesteps, time_notes=time_notes)
    return state, update_by_phase, update_frame


if __name__ == "__main__":
    args = parse_args()

    run_path = Path(args.run_path)
    name = run_path.name.split(".")[0]
    with open(run_path) as f:
        timesteps = json.loads(f.read())

    plt.close("all")
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)

    state, update_by_phase, update_frame = setup_from_timesteps(
        timesteps, fig, time_notes=not args.last_only
    )

    if args.last_only:
        # Advance through all timesteps and phases so the final state is fully populated
        for t in range(len(timesteps)):
            for phase in Phases:
                update_by_phase(t, phase)
        fig.suptitle("")

        png_path = run_path.parent / f"{run_path.parent.name.replace('_', '')}.png"
        state.axes.fig.savefig(str(png_path), dpi=180)
        plt.close(state.axes.fig)
        print(png_path)
    else:
        # Full animation
        total_frames = len(timesteps) * FRAMES_PER_TIMESTEP
        anim = FuncAnimation(
            state.axes.fig, update_frame, frames=total_frames, interval=600, blit=False
        )

        # # Save GIF
        # gif_path = directory / "dashboard.gif"
        # anim.save(str(gif_path), writer=PillowWriter(fps=2))

        # Save MP4 if possible
        mp4_path = run_path.parent / f"{run_path.parent.name.replace('_', '')}.mp4"
        # try:
        writer = FFMpegWriter(fps=args.fps, bitrate=args.bitrate)
        anim.save(str(mp4_path), writer=writer)
        # except Exception:
        #     mp4_path = None

        plt.close(state.axes.fig)
        print(mp4_path)
