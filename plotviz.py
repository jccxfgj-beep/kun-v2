"""Visualize the results of every sweep defined in `experiment.py`.

Reads `results/<sweep>.jsonl` (one JSON line per run, schema in PROTOCOL §7),
groups by config + averages over the 5 seeds, and emits:

  - PNG figures under `figures/` (matplotlib).
  - Markdown data tables under `results/` for the data that backs each figure
    (so the numbers are inspectable without re-parsing JSONL).

Each `make_<sweep>` function is independent so you can re-render one figure
without touching the others. CLI dispatches by sweep name; `all` does every
sweep that has a results file present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_results(path: Path) -> List[Dict[str, Any]]:
    """Read sweep results, auto-detecting .json (list) vs .jsonl (one per line).

    If the requested path is missing, fall back to the sibling with the
    alternate suffix so a sweep can be written as either format without
    forcing every call site to change.
    """
    if not path.exists():
        alt = path.with_suffix(".json" if path.suffix == ".jsonl" else ".jsonl")
        if alt.exists():
            path = alt
        else:
            print(f"[plotviz] skip: {path} not found", file=sys.stderr)
            return []
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# Backward-compat alias: existing callers / external scripts can keep using this name.
read_jsonl = read_results


def mean_std(xs: Iterable[float]) -> Tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def group_by(rows: List[Dict[str, Any]], *keys: str) -> Dict[Tuple, List[Dict]]:
    out: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def write_md_table(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[plotviz] wrote {path}")


def save_fig(path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plotviz] wrote {path}")


# ---------------------------------------------------------------------------
# Per-sweep renderers
# ---------------------------------------------------------------------------

def make_complexity_scaling() -> None:
    rows = read_jsonl(RESULTS_DIR / "complexity_scaling.jsonl")
    if not rows:
        return

    by_ndim_elem = group_by(rows, "model", "shape")
    points: Dict[Tuple[str, int], List[Tuple[int, float, float]]] = defaultdict(list)
    for (model, shape_list), group in by_ndim_elem.items():
        shape = tuple(shape_list) if isinstance(shape_list, list) else shape_list
        n_elems = reduce(mul, shape, 1)
        ndim = len(shape)
        t_mean, t_std = mean_std([r["train_time_s"] for r in group])
        points[(model, ndim)].append((n_elems, t_mean, t_std))

    plt.figure(figsize=(6, 4.5))
    for (model, ndim), pts in sorted(points.items(), key=lambda x: x[0][1]):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        label = f"{model} ND={ndim}"
        plt.errorbar(xs, ys, yerr=errs, marker="o", label=label, capsize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\prod_i D_i$  (total element count)")
    plt.ylabel("train_time_s (mean, log)")
    plt.title("Complexity scaling — walltime")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    save_fig(FIGURES_DIR / "fig_complexity_walltime.png")

    table_rows = []
    for (model, ndim), pts in sorted(points.items(), key=lambda x: x[0][1]):
        for (n_elems, t_mean, t_std) in sorted(pts):
            table_rows.append([model, ndim, n_elems, f"{t_mean:.4g}", f"{t_std:.2g}"])
    write_md_table(
        RESULTS_DIR / "complexity_scaling.md",
        ["model", "ndim", "elements", "train_time_s_mean", "train_time_s_std"],
        table_rows,
    )


def make_five_d_scaling() -> None:
    rows = read_jsonl(RESULTS_DIR / "five_d_scaling.jsonl")
    if not rows:
        return

    by_shape = group_by(rows, "shape")
    points: List[Tuple[int, float, float, int]] = []
    for shape_list, group in by_shape.items():
        shape = tuple(shape_list) if isinstance(shape_list, list) else shape_list
        n_elems = reduce(mul, shape, 1)
        t_mean, t_std = mean_std([r["train_time_s"] for r in group])
        D = shape[1] if len(shape) >= 2 else shape[0]
        points.append((n_elems, t_mean, t_std, D))
    points.sort()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [p[0] for p in points]
    ts = [p[1] for p in points]
    terr = [p[2] for p in points]
    ax.errorbar(xs, ts, yerr=terr, marker="o", color="C0", label="train_time_s", capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\prod_i D_i = 64 \cdot D^4$")
    ax.set_ylabel("train_time_s (s)")
    plt.title("5D scaling — wall-clock vs element count")
    save_fig(FIGURES_DIR / "fig_5d_scaling.png")

    write_md_table(
        RESULTS_DIR / "five_d_scaling.md",
        ["D (spatial)", "elements", "train_time_s_mean", "train_time_s_std"],
        [[p[3], p[0], f"{p[1]:.4g}", f"{p[2]:.2g}"] for p in points],
    )


def make_depth_kernel_scaling() -> None:
    rows = read_jsonl(RESULTS_DIR / "depth_kernel_scaling.jsonl")
    if not rows:
        return

    by_dk = defaultdict(list)
    for r in rows:
        by_dk[(int(r["depth"]), r["kernel"])].append(r)

    summary: Dict[str, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for (depth, kernel), group in by_dk.items():
        mse_mean, mse_std = mean_std([r["best_test_masked_mse"] for r in group])
        t_mean, t_std = mean_std([r["train_time_s"] for r in group])
        summary[kernel].append((depth, mse_mean, mse_std, t_mean, t_std))

    # (a) log-log MSE vs depth
    plt.figure(figsize=(6, 4.5))
    for kernel, pts in summary.items():
        pts.sort()
        ds = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        es = [p[2] for p in pts]
        plt.errorbar(ds, ms, yerr=es, marker="o", label=kernel, capsize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("stack depth")
    plt.ylabel("best test masked-MSE (mean over 5 seeds)")
    plt.title("Depth scaling law — MSE vs depth")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    save_fig(FIGURES_DIR / "fig_depth_scaling.png")

    # (b) wall-clock vs MSE Pareto
    plt.figure(figsize=(6, 4.5))
    for kernel, pts in summary.items():
        pts.sort()
        ts = [p[3] for p in pts]
        ms = [p[1] for p in pts]
        plt.plot(ts, ms, marker="o", label=kernel)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("train_time_s (mean)")
    plt.ylabel("best test masked-MSE (mean)")
    plt.title("Depth–MSE Pareto front by kernel")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    save_fig(FIGURES_DIR / "fig_depth_pareto.png")

    flat_rows = []
    for kernel, pts in summary.items():
        for (depth, mse_m, mse_s, t_m, t_s) in sorted(pts):
            flat_rows.append([kernel, depth, f"{mse_m:.4g}", f"{mse_s:.2g}",
                              f"{t_m:.3g}", f"{t_s:.2g}"])
    write_md_table(
        RESULTS_DIR / "depth_kernel_scaling.md",
        ["kernel", "depth", "mse_mean", "mse_std", "train_time_s_mean", "train_time_s_std"],
        flat_rows,
    )


def _render_main_table(sweep: str) -> None:
    """Markdown-only cross-Ndim table (mean ± std MSE, mean wall-clock).

    Shared by `main_results` (KUN + baselines) and `baselines_main`
    (baseline-only); only the source/destination file name differs.
    """
    rows = read_jsonl(RESULTS_DIR / f"{sweep}.jsonl")
    if not rows:
        return

    # Group by `dataset` too: ETT runs put four distinct datasets at the same
    # (ndim, hidden_dim), so without this they would collapse into one averaged
    # row. The `dataset` column is only emitted when more than one is present
    # (so synthetic-only sweeps keep their original columns).
    by_cell = group_by(rows, "dataset", "model", "ndim", "hidden_dim")
    show_ds = len({ds for (ds, _m, _n, _h) in by_cell.keys()}) > 1
    table = []
    for (ds, model, ndim, hidden), group in by_cell.items():
        mse_m, mse_s = mean_std([r["best_test_masked_mse"] for r in group])
        t_m, _ = mean_std([r["train_time_s"] for r in group])
        gpu_vals = [r["gpu_peak_alloc_mb"] for r in group
                    if r.get("gpu_peak_alloc_mb") is not None]
        gpu_str = f"{(sum(gpu_vals) / len(gpu_vals)):.0f}" if gpu_vals else "—"
        bs_set = sorted({r.get("batch_size") for r in group if r.get("batch_size") is not None})
        if not bs_set:
            bs_str = "—"
        elif len(bs_set) == 1:
            bs_str = str(bs_set[0])
        else:
            bs_str = f"{bs_set[0]}–{bs_set[-1]}"
        table.append([ds, ndim, model, hidden, bs_str, f"{mse_m:.4g} ± {mse_s:.2g}",
                      f"{t_m:.3g}", gpu_str, len(group)])
    table.sort(key=lambda r: (str(r[0]), r[1], r[2], r[3] or 0))
    header = ["ndim", "model", "hidden_dim", "batch_size", "MSE (mean ± std)",
              "train_time_s", "gpu_peak_alloc_mb", "n_seeds"]
    if show_ds:
        header = ["dataset"] + header
    else:
        table = [row[1:] for row in table]
    write_md_table(RESULTS_DIR / f"{sweep}.md", header, table)


def make_main_results() -> None:
    """Markdown-only: cross-Ndim KUN vs baselines (mean ± std MSE, mean wall-clock)."""
    _render_main_table("main_results")


def make_baselines_main() -> None:
    """Markdown-only: cross-Ndim baseline-only table (5 baselines × Ndim 1–4)."""
    _render_main_table("baselines_main")


def make_spec_3d_ablation() -> None:
    rows = read_jsonl(RESULTS_DIR / "spec_3d_ablation.jsonl")
    if not rows:
        return
    by_spec = group_by(rows, "spec")
    table = []
    for (spec,), group in by_spec.items():
        mse_m, mse_s = mean_std([r["best_test_masked_mse"] for r in group])
        t_m, _ = mean_std([r["train_time_s"] for r in group])
        table.append([spec, f"{mse_m:.4g} ± {mse_s:.2g}", f"{t_m:.3g}", len(group)])
    table.sort(key=lambda r: float(r[1].split()[0]) if r[1] != "nan ± nan" else 1e9)
    write_md_table(
        RESULTS_DIR / "spec_3d_ablation.md",
        ["spec", "MSE (mean ± std)", "train_time_s", "n_seeds"],
        table,
    )


def make_spec_3d_permutation() -> None:
    rows = read_jsonl(RESULTS_DIR / "spec_3d_permutation.jsonl")
    if not rows:
        return
    # Discover the per-category schema from the first row that has it -- keeps
    # the renderer working both with category-aware runs and legacy runs
    # without test_mse_by_category populated.
    cat_names: List[str] = []
    for r in rows:
        cats = r.get("test_mse_by_category") or {}
        if cats:
            cat_names = list(cats.keys())
            break
    by_spec = group_by(rows, "spec")
    table = []
    for (spec,), group in by_spec.items():
        mse_m, mse_s = mean_std([r["best_test_masked_mse"] for r in group])
        t_m, _ = mean_std([r["train_time_s"] for r in group])
        row = [spec, f"{mse_m:.4g} ± {mse_s:.2g}", f"{t_m:.3g}", len(group)]
        for cn in cat_names:
            cm, cs = mean_std([
                (r.get("test_mse_by_category") or {}).get(cn) for r in group
            ])
            row.append(f"{cm:.4g} ± {cs:.2g}")
        table.append(row)
    table.sort(key=lambda r: float(r[1].split()[0]) if r[1] != "nan ± nan" else 1e9)
    header = ["spec", "MSE (mean ± std)", "train_time_s", "n_seeds"] + \
             [f"MSE[{cn}]" for cn in cat_names]
    write_md_table(
        RESULTS_DIR / "spec_3d_permutation.md",
        header,
        table,
    )


def make_cross_dim_ablation() -> None:
    rows = read_jsonl(RESULTS_DIR / "cross_dim_ablation.jsonl")
    if not rows:
        return
    by_cfg = group_by(rows, "config")
    table = []
    for (cfg_name,), group in by_cfg.items():
        # strip seed suffix to make the row label
        label = cfg_name.rsplit("_seed", 1)[0]
        mse_m, mse_s = mean_std([r["best_test_masked_mse"] for r in group])
        t_m, _ = mean_std([r["train_time_s"] for r in group])
        table.append([label, f"{mse_m:.4g} ± {mse_s:.2g}", f"{t_m:.3g}", len(group)])
    table.sort()
    write_md_table(
        RESULTS_DIR / "cross_dim_ablation.md",
        ["variant", "MSE (mean ± std)", "train_time_s", "n_seeds"],
        table,
    )


RENDERERS = {
    "main_results":         make_main_results,
    "baselines_main":       make_baselines_main,
    "complexity_scaling":   make_complexity_scaling,
    "spec_3d_ablation":     make_spec_3d_ablation,
    "spec_3d_permutation":  make_spec_3d_permutation,
    "cross_dim_ablation":   make_cross_dim_ablation,
    "five_d_scaling":       make_five_d_scaling,
    "depth_kernel_scaling": make_depth_kernel_scaling,
}


def main():
    global RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=list(RENDERERS.keys()) + ["all"])
    ap.add_argument("--results-dir", default=None,
                    help="directory holding <sweep>.jsonl and where .md tables "
                         "are written (default: ./results next to this script)")
    args = ap.parse_args()
    if args.results_dir is not None:
        RESULTS_DIR = Path(args.results_dir)
    targets = RENDERERS.keys() if args.sweep == "all" else [args.sweep]
    for name in targets:
        RENDERERS[name]()


if __name__ == "__main__":
    main()
