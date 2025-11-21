#!/usr/bin/env python3
"""
Build a conversations.html from a run-path of Member* folders containing conversation_*.json files.

Usage:
    python build_conversations.py /path/to/run-path
"""

import argparse
import colorsys
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import seaborn as sns

FILENAME_ROUNDS_RE = re.compile(r"conversation_(?:\d+_)?rounds_(\d+)(?:-\d+)?\.json$", re.IGNORECASE)


def assign_member_colors(agents: list[str], target_lightness: float = 0.6) -> dict[str, str]:
    """
    Generate a consistent, visually balanced color for each agent based on the Seaborn 'rocket' palette.
    Adjusts all colors to have roughly equal lightness so none are too dark on a dark background.

    Args:
        agents: list of agent/member names.
        target_lightness: desired lightness value (0–1). 0.5–0.7 works well for dark backgrounds.

    Returns:
        Dict mapping agent name -> hex color string.
    """
    base_palette = sns.color_palette("rocket", len(agents) or 1)

    colors = {}
    for agent, (r, g, b) in zip(agents, base_palette):
        # Convert RGB (0–1) to HLS
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        # Normalize lightness toward target
        l = target_lightness
        # Optionally bump saturation a little for vibrancy
        s = min(s * 1.1, 1.0)
        # Convert back to RGB
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        colors[agent] = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"

    return colors




def pretty_maybe_json(text: str) -> str:
    """
    If `text` looks like JSON (starts with '{' or '[') and parses,
    return a pretty-printed version. Otherwise return the original text.
    """
    if not isinstance(text, str):
        return text  # leave non-strings alone

    s = text.strip()
    if not s or s[0] not in "{[":
        return text

    try:
        obj = json.loads(s)
    except Exception:
        return text  # not actually JSON
    else:
        return json.dumps(obj, ensure_ascii=False, indent=2)


def extract_start_round(filename: str) -> Optional[int]:
    m = FILENAME_ROUNDS_RE.search(filename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def get_env_text_first(entry: Dict[str, Any]) -> str:
    """
    First entry example has:
      entry["args"]["items"][0]["content"] -> "TEXT1"
    """
    try:
        items = entry.get("args", {}).get("items", [])
        if items and isinstance(items, list):
            # Some schemas wrap content differently; normalize to string
            content = items[0].get("content", "")
            return str(content)
    except Exception:
        pass
    return ""

def get_env_text_general(entry: Dict[str, Any]) -> str:
    """
    Subsequent entries example has:
      entry["args"]["input"][0]["content"] -> list of {type: "input_text", text: "..."}
    We join all input_text pieces with blank lines.
    """
    try:
        inputs = entry.get("args", {}).get("input", [])
        if inputs and isinstance(inputs, list):
            content_list = inputs[0].get("content", [])
            parts = []
            for c in content_list or []:
                if isinstance(c, dict) and c.get("type") in ("input_text", "text"):
                    parts.append(str(c.get("text", "")))
                elif isinstance(c, str):
                    parts.append(c)
            return "\n\n".join(p for p in parts if p)
    except Exception:
        pass
    return ""

def get_output_text(entry: Dict[str, Any]) -> Optional[str]:
    out = entry.get("output", None)
    if out is None:
        return None
    return str(out)


def parse_conversation_json(path: Path) -> List[Dict[str, Any]]:
    """
    Parses the new schema where each entry is:
      { "role": "system"|"user"|"assistant", "content": str, "reasoning": optional, "phase": str, "time": str }

    Returns a normalized list:
      [
        { "phase": str, "role": str, "content": str, "reasoning": str|None, "time": str|None }
      ]
    """
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON: {path} ({e})")

    turns = []
    for entry in data:
        role = entry.get("role", "")
        content = str(entry.get("content", "")).strip()
        reasoning = entry.get("reasoning", None)
        phase = entry.get("phase", "")
        time = entry.get("time", None)

        turns.append(
            {
                "role": role,
                "content": content,
                "reasoning": reasoning,
                "phase": phase,
                "time": time,
                "model": entry.get("model", None),
                "guard_model": entry.get("guard_model", None),
                "guard_prompt": entry.get("guard_prompt", None),
                "snitch_report": entry.get("snitch_report", None),
            }
        )

    return turns


def build_all_members_timeline(members_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten all assistant messages with timestamps from all members."""
    combined = []

    for member, convs in members_data.items():
        if member == "All Members":
            continue

        for conv in convs:
            for t in conv["turns"]:
                if t.get("role") == "assistant" and t.get("time"):
                    combined.append({
                        "member": member,
                        "time": t["time"],
                        "content": t["content"],
                        "phase": t.get("phase", ""),
                        "reasoning": t.get("reasoning", None),
                        "model": t.get("model", None),
                        "snitch_report": t.get("snitch_report", None),       # <-- NEW
                    })

    # Sort by ISO date/time
    combined.sort(key=lambda x: datetime.strptime(x["time"], "%d %b %y %I:%M:%S %p"))
    return combined


def build_data(run_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
    {
      "Member1": [
         {"file": "conversation_1_rounds_1-2.json", "start_round": 1, "turns": [...]},
         ...
      ],
      ...
    }
    """

    def extract_conv_index(filename: str) -> int:
        # Parse the integer after 'conversation_'
        m = re.match(r"conversation_(\d+)", filename, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return 999999999  # fallback to push unknowns to end

    members: Dict[str, List[Dict[str, Any]]] = {}

    for child in sorted(run_path.iterdir()):
        if child.is_dir() and child.name.startswith("Member"):
            convs = []

            # ✅ force numeric sort rather than string sort
            json_files = sorted(child.glob("conversation_*.json"), key=lambda p: extract_conv_index(p.name))

            for json_file in json_files:
                start_round = extract_start_round(json_file.name)
                turns = parse_conversation_json(json_file)

                # ✅ propagate model from first entry to all turns
                model = turns[0].get("model") if turns else None
                if model:
                    for t in turns:
                        if not t.get("model"):
                            t["model"] = model

                convs.append(
                    {
                        "file": json_file.name,
                        "start_round": start_round,
                        "turns": turns,
                    }
                )

            members[child.name] = convs

    return members


def html_page(
    members_data: Dict[str, List[Dict[str, Any]]],
    member_colors: Dict[str, str],
    title: str = "Conversations"
) -> str:
    data_json = json.dumps(members_data, ensure_ascii=False)
    colors_json = json.dumps(member_colors, ensure_ascii=False)

    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        f"  <title>{escape(title)}</title>\n"
        "  <style>\n"
        "    html, body { margin:0; padding:0; background:#1e1e1e; color:#ffffff; font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; }\n"
        "    .container { max-width:1200px; margin:0 auto; padding:20px 24px 40px; }\n"
        "    .header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:16px; }\n"
        "    .title { font-size:18px; font-weight:600; margin-right:8px; }\n"
        "    .controls { display:flex; gap:10px; align-items:center; }\n"
        "    label { color:#a3a3a3; font-size:12px; margin-right:6px; }\n"
        "    select { background:#111111; color:#ffffff; border:1px solid #3a3a3a; border-radius:8px; padding:8px 10px; }\n"
        "    .round-pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#2a2a2a; color:#a3a3a3; font-size:12px; border:1px solid #3a3a3a; margin-left:auto; }\n"
        "    .conversation { display:flex; flex-direction:column; gap:16px; }\n"
        "    .turn { display:flex; flex-direction:column; gap:6px; }\n"
        "    .phase { color:#a3a3a3; font-size:12px; margin-bottom:2px; }\n"
        "    .system { color:#a3a3a3; white-space:pre-wrap; line-height:1.5; }\n"
        "    .user { color:#ffffff; white-space:pre-wrap; line-height:1.5; }\n"
        "    .assistant { white-space:pre-wrap; line-height:1.5; }\n"
        "    .empty { color:#a3a3a3; font-style:italic; }\n"
        "    .footer-note { margin-top:24px; color:#a3a3a3; font-size:12px; }\n"
        "    .reasoning { color:#b0b0b0; font-size:13px; margin-top:6px; white-space:pre-wrap; line-height:1.5; display:none; }\n"
        "    .reasoning-toggle { cursor:pointer; color:#7dd3fc; font-size:12px; user-select:none; margin-top:4px; }\n"
        "    .reasoning-toggle:hover { text-decoration:underline; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"container\">\n"
        "    <div class=\"header\">\n"
        "      <div class=\"title\">Conversations</div>\n"
        "      <div class=\"controls\">\n"
        "        <label for=\"memberSelect\">Member</label>\n"
        "        <select id=\"memberSelect\"></select>\n"
        "        <label for=\"convSelect\">Conversation</label>\n"
        "        <select id=\"convSelect\" disabled></select>\n"
        "      </div>\n"
        "      <div class=\"round-pill\" id=\"roundPill\" style=\"display:none;\"></div>\n"
        "    </div>\n"
        "    <section id=\"conversation\" class=\"conversation\"></section>\n"
        "    <div class=\"footer-note\" id=\"footerNote\"></div>\n"
        "  </div>\n"
        "  <script>\n"
        f"    const DATA = {data_json};\n"
        f"    const MEMBER_COLORS = {colors_json};\n"
        "\n"
        "    const memberSelect = document.getElementById('memberSelect');\n"
        "    const convSelect = document.getElementById('convSelect');\n"
        "    const roundPill = document.getElementById('roundPill');\n"
        "    const conversationEl = document.getElementById('conversation');\n"
        "    const footerNote = document.getElementById('footerNote');\n"
        "\n"
        "    let activeMember = null;\n"
        "    let activeConvIdx = 0;\n"
        "\n"
        "    // Helper: render guard prompt/model blocks\n"
        "    function addGuardBlock({ first, member }, container) {\n"
        "      const guardBlock = document.createElement('div');\n"
        "      guardBlock.style.marginBottom = '16px';\n"
        "      guardBlock.style.padding = '10px 12px';\n"
        "      guardBlock.style.background = '#111';\n"
        "      guardBlock.style.border = '1px solid #333';\n"
        "      guardBlock.style.borderRadius = '8px';\n"
        "\n"
        "      if (member) {\n"
        "        const label = document.createElement('div');\n"
        "        label.style.fontSize = '12px';\n"
        "        label.style.color = MEMBER_COLORS[member] || '#7dd3fc';\n"
        "        label.textContent = `[${member}]`;\n"
        "        guardBlock.appendChild(label);\n"
        "      }\n"
        "\n"
        "      if (first.guard_model) {\n"
        "        const gm = document.createElement('div');\n"
        "        gm.style.fontSize = '12px'; gm.style.color = '#7dd3fc';\n"
        "        gm.textContent = `Guard model: ${first.guard_model}`;\n"
        "        guardBlock.appendChild(gm);\n"
        "      }\n"
        "\n"
        "      if (first.guard_prompt) {\n"
        "        const toggle = document.createElement('div');\n"
        "        toggle.className = 'reasoning-toggle'; toggle.textContent = '▶ Show guard prompt';\n"
        "\n"
        "        const gp = document.createElement('div');\n"
        "        gp.className = 'reasoning'; gp.textContent = first.guard_prompt; gp.style.display = 'none';\n"
        "\n"
        "        toggle.addEventListener('click', () => {\n"
        "          const visible = gp.style.display === 'block';\n"
        "          gp.style.display = visible ? 'none' : 'block';\n"
        "          toggle.textContent = visible ? '▶ Show guard prompt' : '▼ Hide guard prompt';\n"
        "        });\n"
        "\n"
        "        guardBlock.appendChild(toggle);\n"
        "        guardBlock.appendChild(gp);\n"
        "      }\n"
        "\n"
        "      container.appendChild(guardBlock);\n"
        "    }\n"
        "\n"
        "    function initMemberSelect() {\n"
        "      const members = Object.keys(DATA).sort();\n"
        "      if (!members.includes('All Members')) members.unshift('All Members');\n"
        "      memberSelect.innerHTML = '';\n"
        "      members.forEach(name => {\n"
        "        const opt = document.createElement('option');\n"
        "        opt.value = name; opt.textContent = name;\n"
        "        memberSelect.appendChild(opt);\n"
        "      });\n"
        "      activeMember = members[0];\n"
        "      memberSelect.value = activeMember;\n"
        "      updateConversationSelector();\n"
        "      renderConversation();\n"
        "    }\n"
        "\n"
        "    memberSelect.onchange = () => {\n"
        "      activeMember = memberSelect.value;\n"
        "      activeConvIdx = 0;\n"
        "      updateConversationSelector();\n"
        "      renderConversation();\n"
        "    };\n"
        "\n"
        "    function updateConversationSelector() {\n"
        "      const convs = DATA[activeMember] || [];\n"
        "      convSelect.innerHTML = '';\n"
        "      if (convs.length === 0) {\n"
        "        convSelect.disabled = true;\n"
        "        roundPill.style.display = 'none';\n"
        "        return;\n"
        "      }\n"
        "      convs.forEach((c, idx) => {\n"
        "        const opt = document.createElement('option');\n"
        "        const sr = (c.start_round !== null && c.start_round !== undefined) ? `start ${c.start_round}` : '';\n"
        "        opt.value = idx;\n"
        "        opt.textContent = `${c.file}${sr ? ' ('+sr+')' : ''}`;\n"
        "        convSelect.appendChild(opt);\n"
        "      });\n"
        "      convSelect.disabled = false;\n"
        "      convSelect.value = String(activeConvIdx);\n"
        "      convSelect.onchange = (e) => { activeConvIdx = parseInt(e.target.value, 10) || 0; renderConversation(); };\n"
        "    }\n"
        "\n"
        "    function renderConversation() {\n"
        "      conversationEl.innerHTML = '';\n"
        "      footerNote.textContent = '';\n"
        "\n"
        "      const convs = DATA[activeMember] || [];\n"
        "      if (convs.length === 0) {\n"
        "        const p = document.createElement('p'); p.className='empty'; p.textContent='No conversations found for this member.'; conversationEl.appendChild(p); return;\n"
        "      }\n"
        "\n"
        "      const conv = convs[activeConvIdx] || {};\n"
        "      const sr = conv.start_round;\n"
        "\n"
        "      if (sr !== null && sr !== undefined) {\n"
        "        roundPill.style.display = '';\n"
        "        roundPill.textContent = `Starting round: ${sr}`;\n"
        "      } else {\n"
        "        roundPill.style.display = 'none';\n"
        "      }\n"
        "\n"
        "      const turns = Array.isArray(conv.turns) ? conv.turns : [];\n"
        "      if (turns.length === 0) {\n"
        "        const p = document.createElement('p'); p.className='empty'; p.textContent='Conversation is empty.'; conversationEl.appendChild(p); return;\n"
        "      }\n"
        "\n"
        "      // ---------- GUARD MODEL & PROMPT ----------\n"
        "      if (activeMember !== 'All Members') {\n"
        "        const first = turns[0];\n"
        "        if (first.guard_model || first.guard_prompt) {\n"
        "          addGuardBlock({ first }, conversationEl);\n"
        "        }\n"
        "      } else {\n"
        "        Object.keys(DATA)\n"
        "          .filter(m => m !== 'All Members')\n"
        "          .forEach(member => {\n"
        "            const convs_m = DATA[member];\n"
        "            if (convs_m && convs_m.length > 0) {\n"
        "              const firstTurn = convs_m[0].turns[0];\n"
        "              if (firstTurn && (firstTurn.guard_model || firstTurn.guard_prompt)) {\n"
        "                addGuardBlock({ first: firstTurn, member }, conversationEl);\n"
        "              }\n"
        "            }\n"
        "          });\n"
        "      }\n"
        "\n"
        "      // ---------- RENDER TURNS ----------\n"
        "      turns.forEach((t) => {\n"
        "        const block = document.createElement('div'); block.className='turn';\n"
        "        const phase = document.createElement('div'); phase.className='phase'; phase.textContent = t.phase || ''; block.appendChild(phase);\n"
        "\n"
        "        const msg = document.createElement('div'); msg.className = t.role; msg.textContent = (t.content || '').trim();\n"
        "\n"
        "        if (t.role === 'assistant') {\n"
        "          if (activeMember === 'All Members' && t.member) {\n"
        "            msg.style.color = MEMBER_COLORS[t.member] || '#93c5fd';\n"
        "          } else if (MEMBER_COLORS[activeMember]) {\n"
        "            msg.style.color = MEMBER_COLORS[activeMember];\n"
        "          } else {\n"
        "            msg.style.color = '#93c5fd';\n"
        "          }\n"
        "        }\n"
        "\n"
        "        if (t.role === 'assistant' && t.model) {\n"
        "          const modelTag = document.createElement('div');\n"
        "          modelTag.style.fontSize = '11px'; modelTag.style.color = MEMBER_COLORS[activeMember] || '#93c5fd';\n"
        "          modelTag.textContent = `(${t.model})`;\n"
        "          block.appendChild(modelTag);\n"
        "        }\n"
        "\n"
        "        if (activeMember === 'All Members' && t.member) {\n"
        "          const tag = document.createElement('div');\n"
        "          tag.style.fontSize = '11px'; tag.style.color = MEMBER_COLORS[t.member] || '#93c5fd';\n"
        "          const modelText = t.model ? ` • ${t.model}` : '';\n"
        "          tag.textContent = `[${t.member}${modelText}]`;\n"
        "          block.appendChild(tag);\n"
        "        }\n"
        "\n"
        "        block.appendChild(msg);\n"
        "\n"
        "        // ---------- REASONING ----------\n"
        "        if (t.reasoning) {\n"
        "          const toggle = document.createElement('div');\n"
        "          toggle.className = 'reasoning-toggle'; toggle.textContent = '▶ Show reasoning';\n"
        "\n"
        "          const reasoning = document.createElement('div');\n"
        "          reasoning.className = 'reasoning'; reasoning.textContent = String(t.reasoning).trim();\n"
        "\n"
        "          toggle.addEventListener('click', () => {\n"
        "            const visible = reasoning.style.display === 'block';\n"
        "            reasoning.style.display = visible ? 'none' : 'block';\n"
        "            toggle.textContent = visible ? '▶ Show reasoning' : '▼ Hide reasoning';\n"
        "          });\n"
        "\n"
        "          block.appendChild(toggle);\n"
        "          block.appendChild(reasoning);\n"
        "        }\n"
        "\n"
        "        // ---------- SNITCH REPORT ----------\n"
        "        if (t.snitch_report) {\n"
        "          const parsed = t.snitch_report.parsed || {};\n"
        "          const terminate = t.snitch_report.terminate;\n"
        "\n"
        "          const toggle = document.createElement('div');\n"
        "          toggle.className = 'reasoning-toggle'; toggle.textContent = '▶ Show snitch report';\n"
        "\n"
        "          const box = document.createElement('div');\n"
        "          box.className = 'reasoning'; box.style.display = 'none';\n"
        "\n"
        "          box.textContent = JSON.stringify({ parsed: parsed, terminate: terminate }, null, 2);\n"
        "\n"
        "          toggle.addEventListener('click', () => {\n"
        "            const visible = box.style.display === 'block';\n"
        "            box.style.display = visible ? 'none' : 'block';\n"
        "            toggle.textContent = visible ? '▶ Show snitch report' : '▼ Hide snitch report';\n"
        "          });\n"
        "\n"
        "          block.appendChild(toggle);\n"
        "          block.appendChild(box);\n"
        "        }\n"
        "\n"
        "        if (t.time) {\n"
        "          const time = document.createElement('div');\n"
        "          time.style.fontSize = '11px'; time.style.color = '#888'; time.style.marginTop = '2px';\n"
        "          time.textContent = t.time;\n"
        "          block.appendChild(time);\n"
        "        }\n"
        "\n"
        "        conversationEl.appendChild(block);\n"
        "      });\n"
        "\n"
        "      footerNote.textContent = `${conv.file} • Use the dropdowns to switch member or conversation.`;\n"
        "    }\n"
        "\n"
        "    initMemberSelect();\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Build conversations HTML from Member* folders.")
    parser.add_argument("--run-path", help="Path containing Member* folders")
    args = parser.parse_args()

    run_path = Path(args.run_path).expanduser().resolve()
    if not run_path.exists() or not run_path.is_dir():
        raise SystemExit(f"run-path does not exist or is not a directory: {run_path}")

    members_data = build_data(run_path)
    timeline = build_all_members_timeline(members_data)
    members_data["All Members"] = [{"file": "timeline", "start_round": None, "turns": timeline}]
    agents = sorted([m for m in members_data.keys() if m != "All Members"])
    member_colors = assign_member_colors(agents)

    html = html_page(members_data, member_colors, title="Conversations")

    out_path = run_path / (run_path.name.replace("_", "") + ".html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
