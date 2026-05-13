"""Dump the full conversation for a given conv_hash from regen_wildchat_20k.jsonl.

Usage:
    python experiments/recon_loss_chat/dump_sample.py <conv_hash>
    python experiments/recon_loss_chat/dump_sample.py <conv_hash> --stdout

Writes experiments/recon_loss_chat/data/sample_<conv_hash>.txt unless --stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

JSONL = Path(__file__).parent / "data" / "regen_wildchat_20k.jsonl"


def find(conv_hash: str) -> dict | None:
    with JSONL.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("conv_hash") == conv_hash:
                return d
    return None


def render(d: dict) -> str:
    out = [
        f"conv_hash: {d['conv_hash']}",
        f"language: {d.get('language')}",
        f"n_user_turns: {d.get('n_user_turns')}  n_asst_turns: {d.get('n_asst_turns')}",
        f"source: {JSONL.name} (gemma-3-12b-it regenerated assistant turns)",
    ]
    for i, m in enumerate(d["messages"]):
        out.append(f"\n===== TURN {i} [{m['role']}] ({len(m['content'])} chars) =====")
        out.append(m["content"])
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("conv_hash")
    p.add_argument("--stdout", action="store_true",
                   help="print to stdout instead of writing to data/")
    args = p.parse_args(argv)

    d = find(args.conv_hash)
    if d is None:
        print(f"conv_hash {args.conv_hash} not found in {JSONL}", file=sys.stderr)
        return 1

    text = render(d)
    if args.stdout:
        sys.stdout.write(text)
    else:
        out_path = JSONL.parent / f"sample_{args.conv_hash}.txt"
        out_path.write_text(text)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
