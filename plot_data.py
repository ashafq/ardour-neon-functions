#!/usr/bin/env python3
import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


FRAME_RE = re.compile(r"^\s*Frame Size:\s*(\d+)\s*(.*)$")


@dataclass
class Row:
    fn: str
    nframe: int
    t_default: float
    t_opt: float
    unit_test: str = ""
    notes: str = ""


def parse_blocks(text: str) -> Dict[int, List[Row]]:
    """
    Parse flat CSV rows like:

      Function,Frame Size,Benchmark Time (default),Benchmark Time (optimized),Unit Test Result,Notes
      compute_peak,5,1.378587e-09,2.634015e-09,PASS,"..."

    Returns: { frame_size: [Row(...), ...], ... }
    """
    blocks: Dict[int, List[Row]] = {}

    # Iterate line-by-line so we can be tolerant of stray lines / comments.
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Optional: skip common non-data lines
        if line.startswith("#"):
            continue

        try:
            parsed = next(csv.reader([line]))
        except Exception:
            continue

        if not parsed:
            continue

        # Skip header row (case-insensitive)
        first = parsed[0].strip().lower()
        if first in {"function", "fn"}:
            continue

        # Need at least: fn, frame_size, t_default, t_opt
        if len(parsed) < 4:
            continue

        fn = parsed[0].strip()

        try:
            frame = int(parsed[1])
            t_def = float(parsed[2])
            t_opt = float(parsed[3])
        except ValueError:
            continue

        unit = parsed[4].strip() if len(parsed) > 4 else ""
        notes = parsed[5].strip() if len(parsed) > 5 else ""

        r = Row(fn=fn, nframe=frame, t_default=t_def, t_opt=t_opt,
                unit_test=unit, notes=notes)

        blocks.setdefault(frame, []).append(r)

    return blocks


def improvement_pct(t_def: float, t_opt: float) -> float:
    """
    Positive means optimized is faster (lower time).
    """
    if t_def == 0.0:
        return 0.0
    return (t_def - t_opt) / t_def * 100.0


def summarize(blocks: Dict[int, List[Row]],
              mode: str = "mean",
              only_pass: bool = True,
              include: Optional[List[str]] = None,
              exclude: Optional[List[str]] = None) -> Tuple[List[int], List[float]]:
    frames = sorted(blocks.keys())
    ys: List[float] = []

    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else None

    for f in frames:
        vals: List[float] = []
        for r in blocks[f]:
            if only_pass and r.unit_test and r.unit_test.upper() != "PASS":
                continue
            if include_set is not None and r.fn not in include_set:
                continue
            if exclude_set is not None and r.fn in exclude_set:
                continue
            vals.append(improvement_pct(r.t_default, r.t_opt))

        if not vals:
            ys.append(0.0)
            continue

        if mode == "mean":
            ys.append(sum(vals) / len(vals))
        elif mode == "median":
            s = sorted(vals)
            mid = len(s) // 2
            ys.append(s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid]))
        else:
            raise ValueError("mode must be 'mean' or 'median'")

    return frames, ys


def plot(frames: List[int],
         ys: List[float],
         title: str,
         outpath: Optional[Path],
         show: bool,
         ymin: Optional[float],
         ymax: Optional[float],
         annotate: bool):
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=100)

    x = range(len(frames))
    ax.bar(x, ys)

    ax.set_title(title)
    ax.set_xlabel("Frame Size")
    ax.set_ylabel("Average Improvement, %")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(f) for f in frames])

    ax.grid(True, axis="both")
    ax.set_axisbelow(True)

    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    if annotate:
        for xi, yi in zip(x, ys):
            ax.text(xi, yi, f"{yi:.0f}", ha="center",
                    va=("bottom" if yi >= 0 else "top"))

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath)
        print(f"Wrote: {outpath}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot average improvement vs frame size from benchmark log.")
    ap.add_argument("input", help="Path to the benchmark output text (or '-' for stdin)")
    ap.add_argument("--title", default="Benchmark Results (Default vs. Optimized)")
    ap.add_argument("--out", default="improvement.png", help="Output PNG path (use '' to disable saving)")
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    ap.add_argument("--mode", choices=["mean", "median"], default="mean")
    ap.add_argument("--all", dest="only_pass", action="store_false", help="Include failing unit tests too")
    ap.add_argument("--include", nargs="*", help="Only include these function names")
    ap.add_argument("--exclude", nargs="*", help="Exclude these function names")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    #ap.add_argument("--annotate", action="store_true", help="Label bars with rounded %")
    args = ap.parse_args()

    if args.input == "-":
        import sys
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8", errors="replace")

    blocks = parse_blocks(text)
    if not blocks:
        raise SystemExit("No frame-size blocks found. Is the input in the expected format?")

    frames, ys = summarize(
        blocks,
        mode=args.mode,
        only_pass=args.only_pass,
        include=args.include,
        exclude=args.exclude,
    )

    outpath = Path(args.out) if args.out else None
    plot(frames, ys, args.title, outpath, args.show, args.ymin, args.ymax, False)


if __name__ == "__main__":
    main()
