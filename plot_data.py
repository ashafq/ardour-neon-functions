#!/usr/bin/env python3
"""
This is some vibe coded stuff. Take this code with a grain of ... vibe.
"""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


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

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        try:
            parsed = next(csv.reader([line]))
        except Exception:
            continue

        if not parsed:
            continue

        first = parsed[0].strip().lower()
        if first in {"function", "fn"}:
            continue

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

        blocks.setdefault(frame, []).append(
            Row(fn=fn, nframe=frame, t_default=t_def, t_opt=t_opt, unit_test=unit, notes=notes)
        )

    return blocks


def improvement_pct(t_def: float, t_opt: float) -> float:
    if t_def == 0.0:
        return 0.0
    return (t_def - t_opt) / t_def * 100.0


def _row_ok(r: Row, only_pass: bool, include: Optional[set], exclude: Optional[set]) -> bool:
    if only_pass and r.unit_test and r.unit_test.upper() != "PASS":
        return False
    if include is not None and r.fn not in include:
        return False
    if exclude is not None and r.fn in exclude:
        return False
    return True


def summarize_by_frame(
    blocks: Dict[int, List[Row]],
    mode: str = "mean",
    only_pass: bool = True,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Tuple[List[int], List[float]]:
    frames = sorted(blocks.keys())
    ys: List[float] = []

    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else None

    for f in frames:
        vals: List[float] = []
        for r in blocks[f]:
            if not _row_ok(r, only_pass, include_set, exclude_set):
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


def summarize_by_function(
    blocks: Dict[int, List[Row]],
    mode: str = "mean",
    only_pass: bool = True,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    frame: Optional[int] = None,
) -> Tuple[List[str], List[float]]:
    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else None

    fn_to_vals: Dict[str, List[float]] = {}

    frames = [frame] if frame is not None else sorted(blocks.keys())
    for f in frames:
        if f not in blocks:
            continue
        for r in blocks[f]:
            if not _row_ok(r, only_pass, include_set, exclude_set):
                continue
            fn_to_vals.setdefault(r.fn, []).append(improvement_pct(r.t_default, r.t_opt))

    fns = sorted(fn_to_vals.keys())
    ys: List[float] = []

    for fn in fns:
        vals = fn_to_vals[fn]
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

    return fns, ys


def plot_simple_bars(
    labels: List[str],
    ys: List[float],
    title: str,
    xlabel: str,
    outpath: Optional[Path],
    show: bool,
    ymin: Optional[float],
    ymax: Optional[float],
    annotate: bool,
    rotate_x: bool = False,
):
    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=120)
    x = range(len(labels))
    ax.bar(x, ys)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Improvement, %")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=(45 if rotate_x else 0),
                       ha=("right" if rotate_x else "center"))

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


def plot_grouped_by_frame(
    blocks: Dict[int, List[Row]],
    title: str,
    outpath: Optional[Path],
    show: bool,
    ymin: Optional[float],
    ymax: Optional[float],
    annotate: bool,
    only_pass: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    mode: str,
):
    """
    X: frame size
    For each frame, plot one bar per function (grouped bars).
    Value: improvement % (or aggregated if duplicates exist).
    """
    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else None

    frames = sorted(blocks.keys())

    # Collect function names that survive filters anywhere
    fn_set = set()
    for f in frames:
        for r in blocks[f]:
            if _row_ok(r, only_pass, include_set, exclude_set):
                fn_set.add(r.fn)
    fns = sorted(fn_set)
    if not fns:
        raise SystemExit("No functions left after filtering.")

    # Build matrix [fn][frame] -> aggregated improvement
    # Handle multiple rows per (fn, frame) by mean/median.
    def agg(vals: List[float]) -> float:
        if not vals:
            return 0.0
        if mode == "mean":
            return sum(vals) / len(vals)
        s = sorted(vals)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])

    fn_frame_vals: Dict[str, Dict[int, float]] = {fn: {} for fn in fns}
    for f in frames:
        per_fn: Dict[str, List[float]] = {}
        for r in blocks[f]:
            if not _row_ok(r, only_pass, include_set, exclude_set):
                continue
            per_fn.setdefault(r.fn, []).append(improvement_pct(r.t_default, r.t_opt))
        for fn in fns:
            fn_frame_vals[fn][f] = agg(per_fn.get(fn, []))

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * len(frames)), 6.0), dpi=120)

    n_frames = len(frames)
    n_fns = len(fns)
    x = list(range(n_frames))

    # bar width per function in a group
    group_width = 0.8
    bar_w = group_width / max(1, n_fns)

    # center bars around each frame tick
    start_offset = -group_width / 2 + bar_w / 2

    for i, fn in enumerate(fns):
        xs = [xi + start_offset + i * bar_w for xi in x]
        ys = [fn_frame_vals[fn][f] for f in frames]
        ax.bar(xs, ys, width=bar_w, label=fn)

        if annotate:
            for xx, yy in zip(xs, ys):
                ax.text(xx, yy, f"{yy:.0f}", ha="center",
                        va=("bottom" if yy >= 0 else "top"), fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Frame Size")
    ax.set_ylabel("Improvement, %")

    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in frames])

    ax.grid(True, axis="both")
    ax.set_axisbelow(True)

    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    # Put legend outside if many functions
    if n_fns <= 6:
        ax.legend()
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath)
        print(f"Wrote: {outpath}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot benchmark improvement (Default vs Optimized).")
    ap.add_argument("input", help="Path to results.txt (or '-' for stdin)")
    ap.add_argument("--title", default=None, help="Custom plot title")
    ap.add_argument("--out", default="improvement.png", help="Output PNG path (use '' to disable saving)")
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    ap.add_argument("--mode", choices=["mean", "median"], default="mean")
    ap.add_argument("--all", dest="only_pass", action="store_false", help="Include failing unit tests too")
    ap.add_argument("--include", nargs="*", help="Only include these function names")
    ap.add_argument("--exclude", nargs="*", help="Exclude these function names")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--annotate", action="store_true", help="Label bars with rounded %")
    ap.add_argument("--by-fn", action="store_true",
                    help="Plot one bar per function (aggregated across frame sizes, or selected with --frame)")
    ap.add_argument("--frame", type=int, default=None,
                    help="When used with --by-fn, restrict aggregation to a single frame size")
    ap.add_argument("--grouped", action="store_true",
                    help="Grouped bars: for each frame size, show one bar per function")

    args = ap.parse_args()

    if args.input == "-":
        import sys
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8", errors="replace")

    blocks = parse_blocks(text)
    if not blocks:
        raise SystemExit("No data rows found. Is the input in the expected CSV format?")

    outpath = Path(args.out) if args.out else None

    if args.grouped:
        title = args.title or "Improvement per Function at Each Frame Size"
        plot_grouped_by_frame(
            blocks=blocks,
            title=title,
            outpath=outpath,
            show=args.show,
            ymin=args.ymin,
            ymax=args.ymax,
            annotate=args.annotate,
            only_pass=args.only_pass,
            include=args.include,
            exclude=args.exclude,
            mode=args.mode,
        )
        return

    if args.by_fn:
        labels, ys = summarize_by_function(
            blocks,
            mode=args.mode,
            only_pass=args.only_pass,
            include=args.include,
            exclude=args.exclude,
            frame=args.frame,
        )
        if args.title is None:
            if args.frame is None:
                title = "Average Improvement per Function (all frame sizes)"
            else:
                title = f"Improvement per Function (frame size = {args.frame})"
        else:
            title = args.title

        plot_simple_bars(
            labels=labels,
            ys=ys,
            title=title,
            xlabel="Function",
            outpath=outpath,
            show=args.show,
            ymin=args.ymin,
            ymax=args.ymax,
            annotate=args.annotate,
            rotate_x=True,
        )
    else:
        frames, ys = summarize_by_frame(
            blocks,
            mode=args.mode,
            only_pass=args.only_pass,
            include=args.include,
            exclude=args.exclude,
        )
        labels = [str(f) for f in frames]
        title = args.title or "Average Improvement vs Frame Size (across functions)"

        plot_simple_bars(
            labels=labels,
            ys=ys,
            title=title,
            xlabel="Frame Size",
            outpath=outpath,
            show=args.show,
            ymin=args.ymin,
            ymax=args.ymax,
            annotate=args.annotate,
            rotate_x=False,
        )


if __name__ == "__main__":
    main()
