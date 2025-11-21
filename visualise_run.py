import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from tqdm import tqdm

from render_conversation import (
    AGT_FONT,
    ENV_FONT,
    build_conversation_timeline,
    build_entry_line_spans,
    draw_conversation_scrolling,
)
from render_dashboard import Phases, setup_from_timesteps
from simulate_cpr import ConversationEvent

EVENT_PHASE_SEQ = {
    ConversationEvent.AUDIT_VOTE: [Phases.POST_CONVERSATION, Phases.POLICY_VOTE],
    ConversationEvent.RESOURCE_EXTRACTION: [
        Phases.POST_VOTE,              # 1st time we see this ENV in a round
        Phases.EXTRACTION,             # 2nd
        Phases.WORLD_POST_EXTRACTION,  # 3rd
        Phases.AGENTS_POST_EXTRACTION  # 4th
    ],
    # ConversationEvent.REPRODUCTION_OPTION: [Phases.AGENTS_POST_REPRODUCTION, Phases.WORLD_GROWTH],
}

# Full phase order (used when padding)
ALL_PHASES = [
    Phases.START,
    Phases.POST_CONVERSATION,
    Phases.POLICY_VOTE,
    Phases.POST_VOTE,
    Phases.EXTRACTION,
    Phases.WORLD_POST_EXTRACTION,
    Phases.AGENTS_POST_EXTRACTION,
    Phases.AGENTS_POST_REPRODUCTION,
    Phases.WORLD_GROWTH,
]


def parse_args():
    parser = argparse.ArgumentParser(description='Process run data')
    parser.add_argument(
        '--run-path',
        type=Path,
        help='Directory path (e.g. runs/run_1)'
    )
    parser.add_argument(
        '--follow-member',
        type=str,
        default="Member1",
        help='Member to follow (default: Member1)'
    )
    parser.add_argument(
        '--lines-per-frame',
        type=int,
        default=5,
        help='Number of lines per frame (default: 5)'
    )
    parser.add_argument(
        '--bitrate',
        type=int,
        default=3000,
        help='Bitrate of the mp4 output (default: 3000)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='Frames per second of the output mp4 (default: 5)'
    )
    return parser.parse_args()


def build_joint_timeline(conversation_stream, *, round_offset: int = 0):
    """
    One joint frame per conversation entry (for the scroller).
    Dashboard updates:
      - Only on ENV entries (ignore AGENT echoes).
      - Timestep derives strictly from ENV 'GENERAL_DISCUSSION: outgoing 0' boundaries.
      - Multi-phase events advance via per-round counters.
    """
    timeline = []
    # IMPORTANT: start so that the first round_sep we see maps to round_offset
    round_idx = round_offset - 1
    
    cycle = list(EVENT_PHASE_SEQ)
    cycle_index = {c.name: i for i, c in enumerate(cycle)}
    
    checkpoint = 0
    first_round = True
    for idx, entry in enumerate(conversation_stream):
        step = {"conv_idx": idx, "dashboard_steps": []}

        etype = entry.get("type", "")
        phase_str = (entry.get("phase") or "").strip()

        print(f"{idx}: {etype}: {phase_str}")

        if etype == "round_sep":
            if not first_round:
                timeline[-1]["dashboard_steps"] += [
                    (round_idx, Phases.AGENTS_POST_REPRODUCTION),
                    (round_idx, Phases.WORLD_GROWTH),
                ]
                print(f"\t{round_idx} {Phases.AGENTS_POST_REPRODUCTION}")
                print(f"\t{round_idx} {Phases.WORLD_GROWTH}")
            first_round = False
            round_idx += 1
            step["dashboard_steps"].append((round_idx, Phases.START))
            checkpoint = 0
            print(f"\t{round_idx} {Phases.START}")
        elif etype == "env":
            # Determine our position in the cycle
            index = cycle_index.get(phase_str, None)
            if index is not None:
                for i in range(checkpoint, index + 1):
                    for e in EVENT_PHASE_SEQ[cycle[i]]:
                        step["dashboard_steps"].append((round_idx, e))
                        print(f"\t{round_idx} {e}")
                checkpoint = index + 1

        # 3) Append the step (always one joint frame per conversation entry)
        timeline.append(step)

    timeline[-1]["dashboard_steps"] += [
        (round_idx, Phases.AGENTS_POST_REPRODUCTION),
        (round_idx, Phases.WORLD_GROWTH),
    ]
    return timeline


def build_dashboard_schedule_from_joint(joint_timeline, entry_spans):
    """
    Build schedule batches only from *real* conversation entries.
    Synthetic timeline steps (added later) are handled by the synthetic
    batch builder and must be ignored here.
    """
    schedule = []
    n = len(entry_spans)

    for step in joint_timeline:
        # ignore synthetic timeline entries here
        if step.get("synthetic"):
            continue

        steps = step.get("dashboard_steps") or []
        if not steps:
            continue

        i = step.get("conv_idx")
        # guard against missing/invalid indices
        if not isinstance(i, int) or i < 0 or i >= n:
            # skip; will be scheduled via the synthetic builder later
            continue

        line_threshold = entry_spans[i]["start_line"]
        schedule.append({"line_threshold": line_threshold, "steps": steps})

    # ensure increasing thresholds
    schedule.sort(key=lambda s: s["line_threshold"])
    return schedule



def agent_alive_rounds(timesteps, member_name):
    alive = []
    for i, ts in enumerate(timesteps):
        alive_now = any(a["name"] == member_name for a in ts["agents_start"]["alive"])
        alive.append(alive_now)
    return alive


def augment_timeline_for_dead_rounds(timeline, timesteps, member_name):
    alive_flags = agent_alive_rounds(timesteps, member_name)
    rounds_in_timeline = {r for step in timeline for (r, _) in step["dashboard_steps"]}
    full_timeline = list(timeline)

    for r, alive in enumerate(alive_flags):
        if not alive or r not in rounds_in_timeline:
            # Synthetic step: one per phase (1 frame per phase)
            fake_steps = [(r, ph) for ph in Phases]
            full_timeline.append({
                "conv_idx": len(full_timeline),
                "dashboard_steps": fake_steps,
                "synthetic": True,
            })
    return full_timeline


def first_alive_round(timesteps, member_name: str) -> int:
    """
    Return 0-based index of the first round where member_name appears in agents_start.alive.
    If never alive, return 0.
    """
    for i, ts in enumerate(timesteps):
        if any(a["name"] == member_name for a in ts.get("agents_start", {}).get("alive", [])):
            return i
    return 0


def classify_rounds(timesteps, member_name):
    """Return (pre_birth, post_death, alive_rounds) for the given member."""
    alive_flags = [
        any(a["name"] == member_name for a in ts.get("agents_start", {}).get("alive", []))
        for ts in timesteps
    ]

    # Find first and last rounds where the member is alive
    first_alive = next((i for i, v in enumerate(alive_flags) if v), 0)
    last_alive = max(
        (i for i, v in enumerate(alive_flags) if v),
        default=len(alive_flags) - 1
    )

    pre_birth = list(range(0, first_alive))
    post_death = list(range(last_alive + 1, len(alive_flags)))
    alive_rounds = list(range(first_alive, last_alive + 1))
    return pre_birth, post_death, alive_rounds


def build_synthetic_schedule_batches_stitched(
    joint_timeline,
    timesteps,
    member_name,
    *,
    lines_per_frame: int,
    total_lines: int,
):
    """
    Build a single, monotonic schedule timeline by:
      - pre-birth synthetic batches first,
      - then conversation (we'll shift it by pre-birth),
      - then post-death & mid-round-suffix batches.

    Returns:
      pre_batches (list of schedule dicts),
      post_batches (list of schedule dicts),
      pre_last (int line-threshold at end of pre-birth),
      final_threshold (int absolute threshold after post tail)
    """
    # classify rounds
    pre_birth, post_death, _ = classify_rounds(timesteps, member_name)

    # Count how many synthetic steps belong to pre-birth and post/mid
    pre_steps, post_steps = [], []

    for step in joint_timeline:
        if not step.get("synthetic") or not step.get("dashboard_steps"):
            continue
        r = step["dashboard_steps"][0][0]
        if r in pre_birth:
            pre_steps.extend(step["dashboard_steps"])
        else:
            # mid-round suffix or post-death
            post_steps.extend(step["dashboard_steps"])

    # --- Pre-birth batches: thresholds start at 0 and grow by lines_per_frame
    pre_batches = []
    pre_last = 0
    for (t, ph) in pre_steps:
        pre_last += lines_per_frame
        pre_batches.append({"line_threshold": pre_last, "steps": [(t, ph)]})

    # --- Post tail starts AFTER conversation part
    # conversation occupies thresholds [pre_last, pre_last + total_lines]
    post_start = pre_last + total_lines

    post_batches = []
    cursor = post_start
    for (t, ph) in post_steps:
        cursor += lines_per_frame
        post_batches.append({"line_threshold": cursor, "steps": [(t, ph)]})

    final_threshold = cursor if post_batches else (pre_last + total_lines)
    return pre_batches, post_batches, pre_last, final_threshold



_PHASE_POS = {ph: i for i, ph in enumerate(ALL_PHASES)}

def pad_round_suffix_phases(joint_timeline):
    """
    For each round, if the timeline scheduled only an early prefix of phases
    (e.g., agent dies mid-round), append a synthetic entry that contains the
    remaining phases for that round in order.
    """
    # Gather phases we already have per round, preserving order
    by_round = {}
    for step in joint_timeline:
        for (r, ph) in step.get("dashboard_steps", []):
            by_round.setdefault(r, []).append(ph)

    # Build synthetic suffix per round if needed
    for r, seen in by_round.items():
        if not seen:
            continue
        last = seen[-1]
        last_idx = _PHASE_POS.get(last, -1)
        if last_idx < len(ALL_PHASES) - 1:
            missing = ALL_PHASES[last_idx + 1 :]
            joint_timeline.append({
                "conv_idx": len(joint_timeline),
                "dashboard_steps": [(r, ph) for ph in missing],
                "synthetic": True,
            })

    return joint_timeline


if __name__ == "__main__":
    # -----------------------------
    # Load data
    # -----------------------------
    args = parse_args()
    directory = args.run_path
    follow_member = args.follow_member
    lines_per_frame = args.lines_per_frame
    bitrate = args.bitrate
    fps = args.fps


    member_directory = directory / follow_member

    conv_files = [(int(f.name.split("_")[1].split("_")[0]), f) for f in member_directory.glob("*.json")]
    conv_files.sort(key=lambda x: x[0])
    conversations = [json.loads(f.read_text()) for _, f in conv_files]

    with open(directory / "main_log.json") as f:
        timesteps = json.loads(f.read())

    # -----------------------------
    # Build conversation message stream
    # -----------------------------
    message_stream = []
    current_round = 0
    for conv_idx, c in enumerate(conversations):
        conv_stream, current_round = build_conversation_timeline(
            c,
            conv_num=conv_idx + 1,
            starting_round=current_round
        )
        message_stream.extend(conv_stream)

    # -----------------------------
    # Figure & layout
    # -----------------------------
    plt.close("all")
    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.4, 0.6])

    dashboard_parent = gs[0]  # top panel
    conversation_parent = gs[1]  # bottom panel

    # Dashboard in the top panel (IMPORTANT: pass dashboard_parent, not fig)
    # state, update_by_phase, _update_frame_unused = setup_from_timesteps(timesteps, fig.add_subplot(dashboard_parent).figure)
    state, update_by_phase, _ = setup_from_timesteps(timesteps, dashboard_parent)

    # Conversation axis in the bottom panel
    ax_conv = fig.add_subplot(conversation_parent)

    # Compute where this member first appears globally (0-based)
    round0 = first_alive_round(timesteps, follow_member)

    # Build the timeline with the correct global round offset
    joint_timeline = build_joint_timeline(message_stream, round_offset=round0)
    # ibreakpoint()

    fig.canvas.draw()
    entry_spans, total_lines = build_entry_line_spans(ax_conv, message_stream, ENV_FONT, AGT_FONT)


    schedule = build_dashboard_schedule_from_joint(joint_timeline, entry_spans)

    entry_spans, total_lines = build_entry_line_spans(ax_conv, message_stream, ENV_FONT, AGT_FONT)
    schedule = build_dashboard_schedule_from_joint(joint_timeline, entry_spans)

    # Augment the timeline for rounds where the followed member is absent/dead
    # (Your function should append timeline entries with "synthetic": True)
    joint_timeline = augment_timeline_for_dead_rounds(joint_timeline, timesteps, follow_member)

    joint_timeline = pad_round_suffix_phases(joint_timeline)

    # # Convert synthetic timeline entries into schedule batches that will fire after
    # # the conversation text has finished scrolling.
    # # Build pre-birth / post-death synthetic batches separately
    # synthetic_batches, pre_last, post_last = build_synthetic_schedule_batches(
    #     joint_timeline,
    #     timesteps,
    #     follow_member,
    #     start_line_threshold=total_lines,
    #     lines_per_frame=lines_per_frame,
    # )
    
    # # How many frames are consumed by pre-birth synthetic steps?
    # pre_birth_frames_offset = math.ceil(pre_last / lines_per_frame) if 'pre_last' in
    # locals() else 0
    
    
    # Build stitched synthetic batches
    pre_batches, post_batches, pre_last, final_threshold = build_synthetic_schedule_batches_stitched(
        joint_timeline,
        timesteps,
        follow_member,
        lines_per_frame=lines_per_frame,
        total_lines=total_lines,
    )

    # SHIFT conversation-driven batches so they start AFTER pre-birth
    for b in schedule:
        b["line_threshold"] += pre_last

    # Add pre-birth (before) and post tail (after), then sort
    schedule = pre_batches + schedule + post_batches
    schedule.sort(key=lambda s: s["line_threshold"])

    # ------------------------------------------------------------
    # Build conversation-driven schedule
    # ------------------------------------------------------------
    entry_spans, total_lines = build_entry_line_spans(ax_conv, message_stream, ENV_FONT, AGT_FONT)
    schedule = build_dashboard_schedule_from_joint(joint_timeline, entry_spans)

    # ------------------------------------------------------------
    # Augment / pad the timeline and stitch synthetic batches
    # ------------------------------------------------------------
    joint_timeline = augment_timeline_for_dead_rounds(joint_timeline, timesteps, follow_member)
    joint_timeline = pad_round_suffix_phases(joint_timeline)

    # Build stitched synthetic batches (pre-birth + post-death)
    pre_batches, post_batches, pre_last, final_threshold = build_synthetic_schedule_batches_stitched(
        joint_timeline,
        timesteps,
        follow_member,
        lines_per_frame=lines_per_frame,
        total_lines=total_lines,
    )

    # Shift conversation-driven schedule thresholds so they start after pre-birth
    for b in schedule:
        b["line_threshold"] += pre_last

    # Combine everything into one unified, monotonic schedule
    schedule = pre_batches + schedule + post_batches
    schedule.sort(key=lambda s: s["line_threshold"])

    # ------------------------------------------------------------
    # Animation frame budget
    # ------------------------------------------------------------
    frames_count = math.ceil(final_threshold / lines_per_frame)
    pre_birth_frames_offset = math.ceil(pre_last / lines_per_frame)

    # ------------------------------------------------------------
    # Init progress and scheduler state
    # ------------------------------------------------------------
    _schedule_ptr = 0  # persistent cursor over the schedule list
    print("-" * 80)
    num_rounds = len(timesteps)
    pbar = tqdm(total=num_rounds * len(Phases))
    stage = f"Round 1; {Phases.START.name}"

    def agent_status_for_frame(frame_idx, lines_per_frame, timesteps, member_name):
        """Return 'pre_birth', 'alive', or 'dead' depending on global round."""
        pre_birth, post_death, _ = classify_rounds(timesteps, member_name)
        current_round = frame_idx // (len(Phases) * lines_per_frame)  # rough proxy
        if current_round in pre_birth:
            return "pre_birth"
        elif current_round in post_death:
            return "dead"
        return "alive"


    def combined_update(frame_idx):
        global _schedule_ptr

        # 1) Global visible lines for this frame
        visible_lines = (frame_idx + 1) * lines_per_frame

        # 2) Fire all scheduled dashboard steps up to current threshold
        while _schedule_ptr < len(schedule) and schedule[_schedule_ptr]["line_threshold"] <= visible_lines:
            for (t, ph) in schedule[_schedule_ptr]["steps"]:
                stage = f"Round {t+1}; {ph.name}"
                pbar.set_description(stage)
                pbar.update()
                update_by_phase(t, ph)
            _schedule_ptr += 1

        # 3) Determine latest fired round for overlay logic
        fired_round = 0
        if _schedule_ptr > 0 and schedule[_schedule_ptr - 1]["steps"]:
            fired_round = schedule[_schedule_ptr - 1]["steps"][-1][0]

        pre_birth, post_death, _ = classify_rounds(timesteps, follow_member)

        # 4) Conversation panel
        ax_conv.cla()
        if fired_round in pre_birth:
            ax_conv.text(0.5, 0.5, "Agent not born yet", ha="center", va="center",
                        fontsize=18, color="gray", alpha=0.7)
            ax_conv.set_axis_off()
        elif fired_round in post_death:
            ax_conv.text(0.5, 0.5, "Agent dead", ha="center", va="center",
                        fontsize=18, color="gray", alpha=0.7)
            ax_conv.set_axis_off()
        else:
            # Shift scroller to start at frame 0 right AFTER pre-birth
            conv_frame_idx = max(frame_idx - pre_birth_frames_offset, 0)
            draw_conversation_scrolling(ax_conv, message_stream, conv_frame_idx,
                                        lines_per_frame=lines_per_frame)

        return []



    anim = FuncAnimation(fig, combined_update, frames=frames_count, interval=200, blit=False)

    out_mp4 = directory / f"cpr_visualisation_{args.follow_member}.mp4"
    try:
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        anim.save(str(out_mp4), writer=writer)
    except Exception:
        pass

    plt.close(fig)
