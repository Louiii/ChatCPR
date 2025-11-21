import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from simulate_cpr import ConversationEvent

# Small text for dense panels
FONT_SIZE_ENV = 7.5
FONT_SIZE_AGENT = 8

# Layout constants (axes coords)
MARGIN_L, MARGIN_R = 0.03, 0.03
MARGIN_T, MARGIN_B = 0.03, 0.03
GAP_BLOCKS = 0.012        # vertical gap between blocks
ENV_FONT  = 7.8
AGT_FONT  = 8.2
ENV_WRAP_CHARS = 110      # approx line width for env text
AGT_WRAP_CHARS = 70       # narrower to mimic 70% width bubble
AGT_WIDTH_FRAC = 0.70     # right-aligned bubble width

# Colors (reuse your mako for consistency)
mako = sns.color_palette("mako", 6)
COLOR_ENV_TXT   = (0.12, 0.12, 0.14)
COLOR_AGENT_BG  = mako[1]             # distinct bubble
COLOR_AGENT_TXT = (1, 1, 1)           # white text on colored bubble
COLOR_BUBBLE_EDGE = mako[3]
BUBBLE_PAD_X = 0.012
BUBBLE_PAD_Y = 0.010


def _recalc_total_height(measured_list):
    th = 0.0
    for j, (_, hh) in enumerate(measured_list):
        th += hh
        if j > 0:
            th += GAP_BLOCKS
    return th


def compute_total_lines(ax, stream):
    # Mirror the same wrapping logic to count lines once up front.
    inner_w_frac = 1.0 - (MARGIN_L + MARGIN_R)
    env_width_frac = inner_w_frac
    bubble_w_frac = AGT_WIDTH_FRAC * inner_w_frac
    bubble_inner_w_frac = bubble_w_frac - 2 * BUBBLE_PAD_X

    env_wrap_chars = _approx_chars_for_width(ax, env_width_frac, ENV_FONT)
    agt_wrap_chars = _approx_chars_for_width(ax, bubble_inner_w_frac, AGT_FONT)

    total = 0
    for entry in stream:
        if entry["type"] == "env":
            total += len(_wrap_lines(entry["text"], env_wrap_chars))
        elif entry["type"] == "agent":
            total += len(_wrap_lines(entry["text"], agt_wrap_chars))
        elif entry["type"] in ["round_sep", "conv_sep"]:
            total += 1  # Separators count as 1 line
    return max(1, total)

def _approx_chars_for_width(ax, width_frac, font_points):
    """
    Approximate how many monospace characters fit across a given fraction of the axes width
    for a given font size. This keeps wrapping consistent with bubble width.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ax_pix_width = ax.get_window_extent(renderer=renderer).width
    # Approximate monospace char width in pixels (~0.60 * font-size points)
    px_per_pt = fig.dpi / 72.0
    char_px = 0.60 * font_points * px_per_pt
    usable_px = max(1, width_frac * ax_pix_width)
    return max(8, int(usable_px // char_px))  # keep a sensible minimum

def _wrap_lines(text, wrap_chars):
    """Return a list of wrapped lines (preserve blanks as empty lines)."""
    lines = []
    for raw in text.splitlines():
        # Preserve empty lines explicitly
        if raw.strip() == "":
            lines.append("")
            continue
        wrapped = textwrap.fill(raw, width=wrap_chars, replace_whitespace=False)
        lines.extend(wrapped.splitlines())
    if not lines:
        lines = [""]
    return lines


def _lines_for_entry(ax, entry, env_font_pt, agt_font_pt):
    # Uses your constants from render_conversation: MARGIN_*, AGT_WIDTH_FRAC, BUBBLE_PAD_X, _approx_chars_for_width, _wrap_lines
    inner_w_frac = 1.0 - (MARGIN_L + MARGIN_R)
    bubble_w_frac = AGT_WIDTH_FRAC * inner_w_frac
    bubble_inner_w_frac = bubble_w_frac - 2 * BUBBLE_PAD_X

    env_wrap = _approx_chars_for_width(ax, inner_w_frac, env_font_pt)
    agt_wrap = _approx_chars_for_width(ax, bubble_inner_w_frac, agt_font_pt)

    t = entry.get("type", "")
    if t in ("round_sep", "conv_sep"):
        return 1
    if t == "env":
        return len(_wrap_lines(entry.get("text", ""), env_wrap))
    if t == "agent":
        return len(_wrap_lines(entry.get("text", ""), agt_wrap))
    return 1


def build_entry_line_spans(ax, stream, env_font_pt, agt_font_pt):
    spans = []
    cursor = 0
    for e in stream:
        n = _lines_for_entry(ax, e, env_font_pt, agt_font_pt)
        n = max(1, n)
        spans.append({"start_line": cursor, "n_lines": n})
        cursor += n
    total_lines = max(1, cursor)
    return spans, total_lines


def _extract_prompt_text(item):
    """
    Handle both shapes:
    - init: item["args"]["items"][0]["content"]
    - normal: item["args"]["input"][0]["content"][0]["text"]
    """
    args = item.get("args", {})
    if "items" in args:
        try:
            return args["items"][0]["content"]
        except Exception:
            return ""
    if "input" in args:
        try:
            return args["input"][0]["content"][0]["text"]
        except Exception:
            return ""
    return ""

def _extract_output_text(item):
    out = item.get("output", "")
    # Keep as string; it can be JSON text which we want to show verbatim
    return "" if out is None else str(out)

def _measure_text_width(text, font_points, ax):
    """Measure the width of text in axes coordinates."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ax_pix_width = ax.get_window_extent(renderer=renderer).width
    px_per_pt = fig.dpi / 72.0
    char_px = 0.60 * font_points * px_per_pt  # monospace approximation
    text_width_chars = max(len(line) for line in text.splitlines()) if text else 0
    text_width_px = text_width_chars * char_px
    return text_width_px / max(ax_pix_width, 1)


# ------------------------------------------------------------------------------------
# TIMELINE EXTRACTION
# ------------------------------------------------------------------------------------
def build_conversation_timeline(conv, conv_num, starting_round=0):
    """
    Build a flat list of message entries using phase information from JSON.
    
    Args:
        conv: The conversation data
        conv_num: The conversation number (for separator display)
        starting_round: The round number to start counting from
        
    Returns: 
        tuple: (frames list, final_round_number)
    """
    frames = []
    current_round = starting_round
    previous_phase = None
    
    # Add conversation separator for subsequent conversations
    if conv_num > 1:
        frames.append({
            'type': 'conv_sep',
            'text': f'Conversation {conv_num}',
            'phase': 'separator'
        })
    
    for i, item in enumerate(conv):
        prm = _extract_prompt_text(item)
        out = _extract_output_text(item)
        phase = item.get('phase', 'UNKNOWN')
        
        # Check for round start - ONLY for GENERAL_DISCUSSION with outgoing 0
        if phase.startswith(ConversationEvent.GENERAL_DISCUSSION.name) and ": outgoing 0" in phase:
            current_round += 1
            # Add round separator before this message
            frames.append({
                'type': 'round_sep',
                'text': f'Round {current_round}',
                'phase': 'separator',
                'round_num': current_round
            })
        
        # Add environment message
        if prm:  # Only add if there's actual prompt text
            frames.append({
                'type': 'env',
                'text': prm,
                'phase': phase,
                'phase_changed': (phase != previous_phase)
            })
        
        # Add agent response if it exists
        if out:
            frames.append({
                'type': 'agent',
                'text': out,
                'phase': phase,
                'phase_changed': False  # Phase change is marked on env message
            })
        
        previous_phase = phase

    return frames, current_round


def draw_conversation_scrolling(ax, stream, frame_idx, lines_per_frame=5):
    """
    Chronological reveal:
      - show first N lines (N grows each frame)
      - if content taller than panel, drop lines from top until it fits
      - BEFORE full: stack messages from TOP → DOWN
      - AFTER full: pin latest content to BOTTOM and push older content up
    """
    ax.cla()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Fractions & sizes
    inner_w_frac = 1.0 - (MARGIN_L + MARGIN_R)
    env_width_frac = inner_w_frac                    # full width for env
    bubble_w_frac = AGT_WIDTH_FRAC * inner_w_frac   # 70% bubble
    bubble_inner_w_frac = bubble_w_frac - 2 * BUBBLE_PAD_X

    # Dynamic wrap widths (derived from actual pixel sizes)
    env_wrap_chars = _approx_chars_for_width(ax, env_width_frac, ENV_FONT)
    agt_wrap_chars = _approx_chars_for_width(ax, bubble_inner_w_frac, AGT_FONT)

    # Convert stream → wrapped blocks with line lists (chronological)
    blocks = []
    for entry in stream:
        if entry["type"] == "env":
            lines = _wrap_lines(entry["text"], env_wrap_chars)
            blocks.append(dict(kind="env", lines=lines))
        elif entry["type"] == "agent":
            lines = _wrap_lines(entry["text"], agt_wrap_chars)
            blocks.append(dict(kind="agent", lines=lines))
        elif entry["type"] == "round_sep":
            blocks.append(dict(kind="round_sep", text=entry["text"]))
        elif entry["type"] == "conv_sep":
            blocks.append(dict(kind="conv_sep", text=entry["text"]))

    # Total lines to reveal by this frame
    total_lines_visible = (frame_idx + 1) * lines_per_frame

    # Take the HEAD of the conversation up to total_lines_visible
    chosen = []
    remaining = total_lines_visible
    for b in blocks:
        if remaining <= 0:
            break
        
        # Handle separators differently
        if b["kind"] in ["round_sep", "conv_sep"]:
            chosen.append(b)
            remaining -= 1  # Separators count as 1 line
        else:
            n = len(b["lines"])
            if n <= remaining:
                chosen.append(dict(kind=b["kind"], lines=b["lines"].copy()))
                remaining -= n
            else:
                # take the first 'remaining' lines of this block
                chosen.append(dict(kind=b["kind"], lines=b["lines"][:remaining]))
                remaining = 0

    # Height helpers
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    def _lines_height(n_lines, font_pt, add_bubble_pad=False):
        # Get height *first* in case it's needed for padding
        ax_pix_height = ax.get_window_extent(renderer=renderer).height

        if n_lines == 0:
            return 0.0  # <-- FIX: Empty block has zero height

        line_px = 1.05 * font_pt * px_per_pt
        total_px = n_lines * line_px  # <-- FIX: Removed max(1, ...)

        if add_bubble_pad:
            total_px += 2 * (BUBBLE_PAD_Y * ax_pix_height)
            
        return total_px / max(ax_pix_height, 1)

    # Measure height and total
    measured = []
    total_h = 0.0
    for i, b in enumerate(chosen):
        if b["kind"] in ["round_sep", "conv_sep"]:
            # Separators have fixed height
            h = 0.05  # Adjust this value as needed
            measured.append((b, h))
        else:
            n_lines = len(b["lines"])
            if b["kind"] == "env":
                h = _lines_height(n_lines, ENV_FONT, add_bubble_pad=False)
            else:
                h = _lines_height(n_lines, AGT_FONT, add_bubble_pad=True)
            measured.append((b, h))
        total_h += h
        if i > 0:
            total_h += GAP_BLOCKS

    available = 1.0 - (MARGIN_T + MARGIN_B)

    # If content is taller than available, drop lines from the TOP until it fits
    overflow = total_h > available and measured
    if overflow:
        while total_h > available and measured:
            b0, h0 = measured[0]
            
            # Skip separators - they can't be partially removed
            if b0["kind"] in ["round_sep", "conv_sep"]:
                measured.pop(0)
                total_h = _recalc_total_height(measured)
                continue
            
            if not b0["lines"]:
                measured.pop(0)
                total_h = _recalc_total_height(measured)
                continue
                
            # remove a single line from the first block
            b0["lines"].pop(0)
            # recompute its height
            n0 = len(b0["lines"])
            if b0["kind"] == "env":
                new_h0 = _lines_height(n0, ENV_FONT, add_bubble_pad=False)
            else:
                new_h0 = _lines_height(n0, AGT_FONT, add_bubble_pad=True)
            measured[0] = (b0, new_h0)
            # if the block emptied, drop it completely
            if n0 == 0:
                measured.pop(0)
            total_h = _recalc_total_height(measured)

    # -----------------------------
    # LAYOUT & DRAW
    # -----------------------------
    if not measured:
        return

    available = 1.0 - (MARGIN_T + MARGIN_B)

    # PHASE 1: Pre-scroll (fill from TOP → DOWN until we hit the bottom)
    if total_h <= available:
        y_top = 1.0 - MARGIN_T  # start at the top margin
        for (b, h) in measured:
            if b["kind"] == "round_sep":
                # Single horizontal line with text
                line_y = y_top - h/2
                ax.axhline(y=line_y, color='gray', linewidth=1, linestyle='-', alpha=0.5)
                ax.text(0.5, line_y, f"  {b['text']}  ", fontsize=10, color='gray',
                        ha='center', va='center', bbox=dict(boxstyle='round', 
                        facecolor='white', edgecolor='none'))
            elif b["kind"] == "conv_sep":
                # Double horizontal line with text  
                line_y1 = y_top - h/3
                line_y2 = y_top - 2*h/3
                ax.axhline(y=line_y1, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
                ax.axhline(y=line_y2, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
                ax.text(0.5, y_top - h/2, f"  {b['text']}  ", fontsize=11, color='black',
                        ha='center', va='center', weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='none'))
            elif b["kind"] == "env":
                if not b.get("lines"):  # Check if lines exist and not empty
                    continue
                # Top-aligned text block
                txt = "\n".join(b["lines"])
                ax.text(
                    MARGIN_L, y_top,
                    txt,
                    fontsize=ENV_FONT, color=COLOR_ENV_TXT,
                    ha="left", va="top", family="monospace"
                )
            else:  # agent
                if not b.get("lines"):  # Check if lines exist and not empty
                    continue
                # Agent bubble
                txt = "\n".join(b["lines"])
                
                # Measure actual text width for dynamic bubble sizing
                text_width = _measure_text_width(txt, AGT_FONT, ax)
                actual_bubble_w = min(bubble_w_frac, text_width + 2 * BUBBLE_PAD_X)
                
                # Right-align: x position is from the right margin
                x0 = 1.0 - MARGIN_R - actual_bubble_w
                rect_y0 = y_top - h
                
                rect = FancyBboxPatch(
                    (x0, rect_y0),
                    actual_bubble_w, h,  # <-- Use actual_bubble_w instead of bubble_w_frac
                    boxstyle="round,pad=0.007,rounding_size=0.02",
                    facecolor=COLOR_AGENT_BG, edgecolor=COLOR_BUBBLE_EDGE, linewidth=1.1
                )
                ax.add_patch(rect)
                
                ax.text(
                    x0 + BUBBLE_PAD_X, y_top - BUBBLE_PAD_Y,
                    txt,
                    fontsize=AGT_FONT, color=COLOR_AGENT_TXT,
                    ha="left", va="top", family="monospace"
                )

            # Move DOWN for the next (older) block
            y_top -= (h + GAP_BLOCKS)

        return  # nothing else to do in pre-scroll

    # PHASE 2: Overflow → bottom-pinned scrolling (your current behavior)
    # The overflow cropping above has already trimmed 'measured' from the head.
    y_bottom = MARGIN_B
    for (b, h) in reversed(measured):
        if b["kind"] == "round_sep":
            # Single horizontal line with text
            line_y = y_bottom + h/2
            ax.axhline(y=line_y, color='gray', linewidth=1, linestyle='-', alpha=0.5)
            ax.text(0.5, line_y, f"  {b['text']}  ", fontsize=10, color='gray',
                    ha='center', va='center', bbox=dict(boxstyle='round', 
                    facecolor='white', edgecolor='none'))
        elif b["kind"] == "conv_sep":
            # Double horizontal line with text
            line_y1 = y_bottom + h/3
            line_y2 = y_bottom + 2*h/3
            ax.axhline(y=line_y1, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
            ax.axhline(y=line_y2, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
            ax.text(0.5, y_bottom + h/2, f"  {b['text']}  ", fontsize=11, color='black',
                    ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='none'))
        elif b["kind"] == "env":
            top_y = y_bottom + h
            txt = "\n".join(b["lines"])
            ax.text(
                MARGIN_L, top_y,
                txt,
                fontsize=ENV_FONT, color=COLOR_ENV_TXT,
                ha="left", va="top", family="monospace"
            )
        else:
            if not b.get("lines"):
                continue
            txt = "\n".join(b["lines"])
            
            # Measure actual text width for dynamic bubble sizing
            text_width = _measure_text_width(txt, AGT_FONT, ax)
            actual_bubble_w = min(bubble_w_frac, text_width + 2 * BUBBLE_PAD_X)
            
            # Right-align: x position is from the right margin
            x0 = 1.0 - MARGIN_R - actual_bubble_w
            rect_y0 = y_bottom
            
            rect = FancyBboxPatch(
                (x0, rect_y0),
                actual_bubble_w, h,  # <-- Use actual_bubble_w instead of bubble_w_frac
                boxstyle="round,pad=0.007,rounding_size=0.02",
                facecolor=COLOR_AGENT_BG, edgecolor=COLOR_BUBBLE_EDGE, linewidth=1.1
            )
            ax.add_patch(rect)
            
            ax.text(
                x0 + BUBBLE_PAD_X, y_bottom + h - BUBBLE_PAD_Y,
                txt,
                fontsize=AGT_FONT, color=COLOR_AGENT_TXT,
                ha="left", va="top", family="monospace"
            )

        y_bottom += (h + GAP_BLOCKS)


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------------------------
    directory = Path("runs/run_19")
    follow_member = "Member1"
    member_directory = directory / follow_member
    conv_files = [(int(f.name.split("_")[1].split("_")[0]), f) for f in member_directory.glob("*.json")]
    conv_files.sort(key=lambda x: x[0])
    conversations = []
    for _, f in conv_files:
        conversations.append(json.loads(f.read_text()))

    conversations = conversations[:2]  # Just for faster rendering.

    # Output files
    gif_path = directory / "conversation_panel.gif"
    mp4_path = directory / "conversation_panel.mp4"

    message_stream = []
    current_round = 0  # Track rounds across all conversations

    for conv_idx, c in enumerate(conversations):
        conv_stream, current_round = build_conversation_timeline(
            c, 
            conv_num=conv_idx+1,
            starting_round=current_round
        )
        message_stream.extend(conv_stream)

    n_frames = len(message_stream)

    # Safety: ensure we always have at least 1 frame
    if n_frames == 0:
        message_stream = [{'type': 'env', 'text': '(no conversation content)', 'phase': 'UNKNOWN'}]
        n_frames = 1

    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    # fig.suptitle("Conversation Thread", fontsize=12)

    # Compute frames from total lines (use same wrap as drawer)
    LINES_PER_FRAME = 4
    FPS = 5
    total_lines = compute_total_lines(ax, message_stream)
    frames_count = math.ceil(total_lines / LINES_PER_FRAME)

    def _update(i):
        draw_conversation_scrolling(ax, message_stream, i, lines_per_frame=LINES_PER_FRAME)
        return []

    anim = FuncAnimation(fig, _update, frames=frames_count, interval=300, blit=False)

    # Save
    anim.save(str(gif_path), writer=PillowWriter(fps=FPS))
    try:
        anim.save(str(mp4_path), writer=FFMpegWriter(fps=FPS, bitrate=1400))
    except Exception:
        pass

    plt.close(fig)
    print("Saved:", gif_path)