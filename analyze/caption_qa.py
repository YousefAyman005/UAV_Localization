"""Quality gate for VLM captions before training on them.

Scans cache/captions/*.jsonl and reports, per file and overall:
  - count + examples of captions that leak color/brightness words (these do NOT
    survive the drone<->satellite gap and undermine the text bridge),
  - count + examples of other banned words (image/satellite/season/...),
  - word-length distribution (target ~25-45),
  - duplicate-caption rate (a VLM stuck on a template is a red flag).

Usage:
    python analyze/caption_qa.py                       # all of cache/captions
    python analyze/caption_qa.py --dir cache/captions --examples 5
"""

import argparse
import collections
import glob
import json
import os
import re

COLOR_RE = re.compile(
    r"\b(?:dark|light|bright|pale|deep|white|black|grey|gray|green|brown|red|"
    r"blue|yellow|orange|golden|tan|beige|silver|reddish|greenish|brownish)"
    r"(?:er|est)?\b", re.IGNORECASE)
BANNED_RE = re.compile(
    r"\b(?:satellite|aerial|drone|photo|image|picture|shadow|shadows|sunlit|"
    r"season|seasonal|lighting|blurry|resolution)\b", re.IGNORECASE)


def scan_file(path, examples):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    caps = [r.get("caption", "") for r in rows]
    n = len(caps)
    if n == 0:
        return None
    color_hits = [(c, COLOR_RE.findall(c)) for c in caps if COLOR_RE.search(c)]
    banned_hits = [(c, BANNED_RE.findall(c)) for c in caps if BANNED_RE.search(c)]
    lengths = [len(c.split()) for c in caps]
    dupes = n - len(set(caps))
    empty = sum(1 for c in caps if not c.strip())
    return {
        "path": path, "n": n,
        "color_n": len(color_hits), "color_ex": color_hits[:examples],
        "banned_n": len(banned_hits), "banned_ex": banned_hits[:examples],
        "len_min": min(lengths), "len_max": max(lengths),
        "len_mean": sum(lengths) / n,
        "len_out": sum(1 for x in lengths if x < 15 or x > 55),
        "dupes": dupes, "empty": empty,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cache/captions")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    if not files:
        print(f"No .jsonl caption files in {args.dir}"); return

    tot = collections.Counter()
    print(f"{'file':40s} {'N':>6} {'color':>7} {'banned':>7} {'dup':>5} "
          f"{'len(min/mean/max)':>20} {'out':>4}")
    print("-" * 95)
    worst = []
    for path in files:
        s = scan_file(path, args.examples)
        if s is None:
            continue
        for k in ("n", "color_n", "banned_n", "dupes", "empty", "len_out"):
            tot[k] += s[k]
        print(f"{os.path.basename(path):40s} {s['n']:>6} "
              f"{s['color_n']:>7} {s['banned_n']:>7} {s['dupes']:>5} "
              f"{s['len_min']:>3}/{s['len_mean']:>4.0f}/{s['len_max']:>3} "
              f"{s['len_out']:>11}")
        worst.append(s)

    n = max(1, tot["n"])
    print("-" * 95)
    print(f"TOTAL {tot['n']} captions | "
          f"color leak {tot['color_n']} ({100*tot['color_n']/n:.1f}%) | "
          f"banned {tot['banned_n']} ({100*tot['banned_n']/n:.1f}%) | "
          f"dupes {tot['dupes']} | empty {tot['empty']} | "
          f"length-outliers {tot['len_out']}")

    # Show a few concrete leaks so the prompt/filter can be tuned.
    for s in worst:
        if s["color_ex"] or s["banned_ex"]:
            print(f"\n  ── {os.path.basename(s['path'])} examples ──")
            for c, hits in s["color_ex"]:
                print(f"    COLOR {hits}: {c}")
            for c, hits in s["banned_ex"]:
                print(f"    BANNED {hits}: {c}")

    leak = tot["color_n"] + tot["banned_n"]
    print(f"\n  Verdict: {'CLEAN — good to train' if leak == 0 else f'{leak} leaks — consider re-running clean or tightening the filter'}")


if __name__ == "__main__":
    main()
