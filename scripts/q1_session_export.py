"""Byte-safe native OpenCode JSON export -> lossless Markdown, not an AI summary.

No group SESSION.md writes. Every exported message is a fenced JSON object;
round-trip equality verifies prompts, assistant content and full tool states.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def render(payload):
    blocks = ["# Authentic OpenCode session transcript\n\nMechanically rendered native export; no message summarization.\n",
              "## Session metadata\n\n```json\n"+json.dumps(payload["info"],ensure_ascii=False,indent=2)+"\n```\n"]
    for i,message in enumerate(payload["messages"],start=1):
        # Variable fence length prevents arbitrary transcript text closing a block.
        text = json.dumps(message,ensure_ascii=False,indent=2)
        fence = "`"*max(4,1+max((len(s) for s in re.findall(r"`+",text)),default=0))
        blocks.append(f"## Message {i}: {message['type']} {message['id']}\n\n{fence}json\n{text}\n{fence}\n")
    return "\n".join(blocks)


def recover(markdown):
    return [json.loads(match.group(2)) for match in re.finditer(r"(?m)^(`{4,})json\n(.*?)\n\1$",markdown,re.S)]


def export(output, *, source=None, session=None, executable=None):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Export must remain inside repository")
    output.mkdir(parents=True,exist_ok=False)
    if source:
        raw = Path(source).read_bytes()
    else:
        if not session or not re.fullmatch(r"ses_[A-Za-z0-9]+",session) or not executable:
            raise ValueError("Supply session ID and native executable")
        completed = subprocess.run([str(executable),"session","export",session],capture_output=True,check=True)
        raw = completed.stdout
    payload = json.loads(raw)
    text = render(payload)
    if recover(text) != payload["messages"]:
        raise ValueError("Markdown round-trip lost message content")
    (output/"session.json").write_bytes(raw)
    (output/"session.md").write_text(text,encoding="utf-8",newline="\n")
    messages = payload["messages"]
    tools = [part for msg in messages if msg["type"] == "assistant" for part in msg.get("content",[]) if part.get("type") == "tool"]
    prompts = [{"id":msg["id"],"heading":msg.get("text","").splitlines()[0][:140]} for msg in messages if msg["type"] == "user"]
    proof = {"session_id":payload["info"]["id"],"message_count":len(messages),"types":dict(Counter(msg["type"] for msg in messages)),
             "tool_parts":len(tools),"tool_states_preserved":all("state" in tool for tool in tools),"round_trip_equal":True,
             "user_prompt_inventory":prompts,"json_sha256":hashlib.sha256(raw).hexdigest(),
             "markdown_sha256":hashlib.sha256((output/"session.md").read_bytes()).hexdigest(),
             "warning":"Live-session snapshot: re-export after the final response/commit. Review and explicitly redact secrets before group submission. Other members/machines are not certified."}
    (output/"export_verification.json").write_text(json.dumps(proof,indent=2,ensure_ascii=False),encoding="utf-8",newline="\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-json",type=Path)
    parser.add_argument("--session")
    parser.add_argument("--opencode-executable",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    proof = export(args.output_dir,source=args.from_json,session=args.session,executable=args.opencode_executable)
    print(json.dumps({key:proof[key] for key in ["session_id","message_count","tool_parts","round_trip_equal"]}))
