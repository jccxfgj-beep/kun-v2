"""Orchestrates the table-producing experiment sweeps.

Each entry point here produces one of the tables in `tables/`:

    main_results       → tab_main_results.tex / tab_synthetic.tex
    baselines_main     → unified main table: any subset of {KUN + 5 baselines}
                          on synthetic ND or ETT (--models / --dataset / --ndims)
    complexity_scaling → tab_complexity.tex
    spec_3d_ablation   → tab_3d_ablation.tex
    cross_dim_ablation → tab_ablation.tex
    five_d_scaling     → 5D anchor figure (training time vs prod D_i at Ndim=5)
    expressiveness     → param-count vs MSE / training-time vs MSE curves

Each sweep is a list of cfg dicts; `train.py` is called for each cfg and the
JSON-lines are accumulated under `results/<sweep_name>.jsonl`.

Heavy/long-running runs should be parallelized at the shell level (one process
per GPU); this orchestrator just enumerates configs and writes them out as
YAML files that the shell scripts under `scripts/` then iterate over.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from data import DEFAULT_CONFIGS

RESULTS_DIR = Path(__file__).parent / "results"

# Generated YAML configs are throwaway intermediates. On Colab the repo lives
# on a Google Drive FUSE mount, which is eventually-consistent — a config
# written here and read back milliseconds later by train.py can fail to open.
# Set KUN_CONFIGS_DIR (the run scripts point it at local disk) to keep configs
# off Drive entirely; unset, it falls back to the in-repo path as before.
CONFIGS_DIR = Path(os.environ.get(
    "KUN_CONFIGS_DIR", Path(__file__).parent / "configs" / "generated"))


# ---------------------------------------------------------------------------
# Sweep generators (each returns a list of cfg dicts to write out)
# ---------------------------------------------------------------------------

SEEDS = [2026, 2016, 2006, 1996, 1986]  # PROTOCOL §6

# Data-generation knobs frozen across every sweep so a given seed produces a
# bit-identical dataset regardless of which model / config path runs. Only
# `seed` is allowed to vary; everything else stays pinned here.
DATA_K = 2
DATA_K_MIN = 1
DATA_N_SAMPLES = 256

DEFAULT_BASELINES_BY_NDIM: Dict[int, List[str]] = {
    1: ["linear", "patchtst"],
    2: ["linear", "patchtst"],
    3: ["convlstm"],
    4: ["mamba4d", "transformer4d"],
}

# Every baseline, now Ndim-generic, used by the `baselines_main` sweep so the
# full baseline x Ndim grid is comparable cell-for-cell against KUN. Includes
# the two neural-operator baselines (fno, uno), dimension-generic by
# construction and run at every Ndim.
ALL_BASELINES: List[str] = ["linear", "patchtst", "convlstm", "mamba4d",
                            "transformer4d", "fno", "uno", "simvp", "predrnn"]

# The unified model menu for the `baselines_main` sweep: KUN plus every
# (Ndim-generic) baseline. `--models` selects any subset of these.
ALL_MODELS: List[str] = ["kun"] + ALL_BASELINES

ETT_DATASETS: List[str] = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

# ETT is a real time-series dataset with a 64-step window. It has two layouts
# only — there is no 3D+ form — so `--dataset ett` accepts Ndim 1 or 2:
#   Ndim 1: univariate, shape (64,)  — each of the 7 columns is its own series.
#   Ndim 2: multivariate, shape (64, 7) — the 7 variables form the second axis.
ETT_SHAPE = (64, 7)          # Ndim-2 (multivariate) layout. spec / m /
                             # shape_name are taken from DEFAULT_CONFIGS so
                             # ETT stays identical to the synthetic 1D/2D format.

# Moving MNIST is a real 3D image-sequence dataset (time x H x W), pre-generated
# by make_mnist_lowres.py at three spatial resolutions. It is 3D only, so
# `--dataset mnist*` always emits Ndim-3 cells. Bare `mnist` -> the default
# resolution; mnist8 / mnist27 / mnist64 select a specific one. 8/27/64 are all
# perfect cubes, so every axis splits cleanly into m=3 chunks.
MNIST_RESOLUTIONS = [8, 27, 64]
MNIST_DEFAULT_RES = 27
MNIST_DATASETS: List[str] = [f"mnist{r}" for r in MNIST_RESOLUTIONS]
# spec / m / shape_name are taken from DEFAULT_CONFIGS[3] — Moving MNIST is 3D
# and uses the same format as the synthetic 3D rows.

SHAPE_NAME_BY_NDIM = {1: "l", 2: "lh", 3: "lhw", 4: "lxyz", 5: "lwxyz",
                      6: "lvwxyz", 7: "luvwxyz", 8: "ltuvwxyz",
                      9: "lstuvwxyz", 10: "lrstuvwxyz"}


def main_results_sweep(
    seeds: List[int] = SEEDS,
    ndims: List[int] = (1, 2, 3, 4),
    baselines_by_ndim: Dict[int, List[str]] = None,
    capacity_grid: List[int] = (32, 64, 128, 256, 512),
    include_kun: bool = True,
    kun_only: bool = False,
    kun_hidden_dim: int = 64,
    kun_latent_dim: int = 64,
    kun_kernel: str = "linear",
    n_epochs: int = 50,
    batch_size_le3: int = 32,
    batch_size_gt3: int = 16,
    lr: float = 1e-3,
) -> List[Dict[str, Any]]:
    """KUN-v2 + per-Ndim baselines at Ndim=1..4 (or 1..5). 5 seeds per cell (PROTOCOL §6).

    Ndim=5 has no per-dim baselines by default; passing it via `ndims` produces
    only the KUN reference cells.
    """
    cfgs = []
    if baselines_by_ndim is None:
        baselines_by_ndim = DEFAULT_BASELINES_BY_NDIM
    if kun_only:
        baselines_by_ndim = {}
    shape_name_by_ndim = {1: "l", 2: "lh", 3: "lhw", 4: "lxyz", 5: "lwxyz",
                          6: "lvwxyz", 7: "luvwxyz", 8: "ltuvwxyz",
                          9: "lstuvwxyz", 10: "lrstuvwxyz"}
    for ndim in ndims:
        if ndim not in DEFAULT_CONFIGS:
            continue
        base_cfg = DEFAULT_CONFIGS[ndim]
        common = dict(
            shape=list(base_cfg["shape"]),
            shape_name=shape_name_by_ndim[ndim],
            m=list(base_cfg["m"]),
            spec=base_cfg["spec"],
            mask_axis=base_cfg["mask_axis"],
            n_epochs=n_epochs,
            batch_size=batch_size_le3 if ndim <= 3 else batch_size_gt3,
            lr=lr,
            K=DATA_K, K_min=DATA_K_MIN, n_samples=DATA_N_SAMPLES,
        )
        for seed in seeds:
            if include_kun:
                cfgs.append({**common, "model": "kun",
                             "hidden_dim": kun_hidden_dim,
                             "latent_dim": kun_latent_dim,
                             "kernel": kun_kernel, "seed": seed,
                             "name": f"kun_nd{ndim}_seed{seed}"})
            for bl in baselines_by_ndim.get(ndim, []):
                for cap in capacity_grid:
                    cfgs.append({**common, "model": bl, "hidden_dim": cap,
                                 "seed": seed,
                                 "name": f"{bl}_nd{ndim}_h{cap}_seed{seed}"})
    return cfgs


def _resolve_datasets(dataset: str) -> List[str]:
    """Map a --dataset value to the concrete dataset name(s) to emit cells for.

    'synthetic' -> the ND sine field (one logical dataset, swept over --ndims).
    'ett'       -> all four ETT files.
    'ETTh1'..   -> that single ETT file.
    'mnist'     -> Moving MNIST at the default resolution (mnist27).
    'mnist8'..  -> Moving MNIST at that specific resolution.
    """
    if dataset == "synthetic":
        return ["synthetic"]
    if dataset == "ett":
        return list(ETT_DATASETS)
    if dataset in ETT_DATASETS:
        return [dataset]
    if dataset == "mnist":
        return [f"mnist{MNIST_DEFAULT_RES}"]
    if dataset in MNIST_DATASETS:
        return [dataset]
    raise ValueError(
        f"unknown --dataset {dataset!r}; expected one of: "
        f"synthetic, ett, {', '.join(ETT_DATASETS)}, "
        f"mnist, {', '.join(MNIST_DATASETS)}"
    )


def baselines_main_sweep(
    seeds: List[int] = SEEDS,
    ndims: List[int] = None,
    models: List[str] = None,
    dataset: str = "synthetic",
    capacity_grid: List[int] = (128,),
    kun_hidden_dim: int = 64,
    kun_latent_dim: int = 64,
    kun_kernel: str = "linear",
    n_epochs: int = 50,
    batch_size_le3: int = 32,
    batch_size_gt3: int = 16,
    lr: float = 1e-3,
    mnist_frames: int = 64,
) -> List[Dict[str, Any]]:
    """Unified main table: any subset of {KUN + 5 baselines} on any dataset.

    Three orthogonal switches (this is the sweep behind run_baselines.sh):

      `models`  -- which models to emit. KUN emits one reference cell per
                   (context, seed); each baseline emits one cell per
                   (context, seed, capacity) over `capacity_grid`.
      `dataset` -- the data source: 'synthetic' (ND sine field), ETT
                   ('ett' = all four files; or a single ETThN/ETTmN), or
                   Moving MNIST ('mnist' = default resolution; or
                   mnist8/mnist27/mnist64).
      `ndims`   -- which Ndim cells to emit. Synthetic supports 1..10;
                   ETT supports only Ndim 1 (univariate) or 2 (multivariate);
                   Moving MNIST supports only Ndim 3. Any other value is an
                   error. When `ndims` is None it defaults to 1..4 for
                   synthetic, [1, 2] for ETT, [3] for Moving MNIST.

    5 seeds per cell by default (PROTOCOL §6).
    """
    if models is None:
        models = list(ALL_MODELS)
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise ValueError(f"unknown model(s) {bad}; known: {ALL_MODELS}")
    include_kun = "kun" in models
    baselines = [m for m in models if m != "kun"]

    datasets = _resolve_datasets(dataset)
    is_synthetic = datasets == ["synthetic"]
    is_mnist = not is_synthetic and all(d.startswith("mnist") for d in datasets)
    is_ett = not is_synthetic and not is_mnist
    if ndims is None:
        if is_mnist:
            ndims = [3]
        elif is_ett:
            ndims = [1, 2]
        else:
            ndims = [1, 2, 3, 4]

    # --- the list of (shape / spec) contexts to emit cells for ---
    contexts: List[Dict[str, Any]] = []
    if is_mnist:
        bad_nd = sorted(set(ndims) - {3})
        if bad_nd:
            raise ValueError(
                f"--dataset {dataset} (Moving MNIST) is a 3D image-sequence "
                f"dataset and supports only Ndim 3; got unsupported Ndim {bad_nd}."
            )
        bc = DEFAULT_CONFIGS[3]
        for ds in datasets:
            res = int(ds[len("mnist"):])
            # `mnist_frames` selects the time axis (the pre-generated .npy is
            # mnist_<res>x<res>_<frames>frame.npy); default 64. The tag carries
            # the frame count only when it differs from the default so existing
            # 64-frame run names / result rows stay unchanged.
            tag = ds if mnist_frames == 64 else f"{ds}f{mnist_frames}"
            contexts.append(dict(
                dataset=ds, ndim=3, tag=tag,
                shape=[mnist_frames, res, res], shape_name=SHAPE_NAME_BY_NDIM[3],
                m=list(bc["m"]), spec=bc["spec"], mask_axis=bc["mask_axis"],
                batch_size=batch_size_le3,
            ))
    elif is_ett:
        bad_nd = sorted(set(ndims) - {1, 2})
        if bad_nd:
            raise ValueError(
                f"--dataset {dataset} (ETT) supports only Ndim 1 (univariate, "
                f"shape (64,)) or Ndim 2 (multivariate, shape (64,7)); got "
                f"unsupported Ndim {bad_nd}."
            )
        for ds in datasets:
            for ndim in sorted(set(ndims)):
                bc = DEFAULT_CONFIGS[ndim]
                # ETT-1D is shape (64,); ETT-2D is (64, 7) (7 variables).
                shape = [64] if ndim == 1 else list(ETT_SHAPE)
                contexts.append(dict(
                    dataset=ds, ndim=ndim, tag=f"{ds}_nd{ndim}",
                    shape=shape, shape_name=SHAPE_NAME_BY_NDIM[ndim],
                    m=list(bc["m"]), spec=bc["spec"], mask_axis=bc["mask_axis"],
                    batch_size=batch_size_le3,
                ))
    else:
        for ndim in ndims:
            if ndim not in DEFAULT_CONFIGS:
                continue
            bc = DEFAULT_CONFIGS[ndim]
            contexts.append(dict(
                dataset="synthetic", ndim=ndim, tag=f"nd{ndim}",
                shape=list(bc["shape"]), shape_name=SHAPE_NAME_BY_NDIM[ndim],
                m=list(bc["m"]), spec=bc["spec"], mask_axis=bc["mask_axis"],
                batch_size=batch_size_le3 if ndim <= 3 else batch_size_gt3,
            ))

    cfgs: List[Dict[str, Any]] = []
    for ctx in contexts:
        common = dict(
            shape=ctx["shape"], shape_name=ctx["shape_name"],
            m=ctx["m"], spec=ctx["spec"], mask_axis=ctx["mask_axis"],
            dataset=ctx["dataset"],
            n_epochs=n_epochs, batch_size=ctx["batch_size"], lr=lr,
        )
        if ctx["dataset"] == "synthetic":
            common.update(K=DATA_K, K_min=DATA_K_MIN, n_samples=DATA_N_SAMPLES)
        for seed in seeds:
            if include_kun:
                cfgs.append({**common, "model": "kun",
                             "hidden_dim": kun_hidden_dim,
                             "latent_dim": kun_latent_dim,
                             "kernel": kun_kernel, "seed": seed,
                             "name": f"kun_{ctx['tag']}_seed{seed}"})
            for bl in baselines:
                for cap in capacity_grid:
                    cfgs.append({**common, "model": bl, "hidden_dim": cap,
                                 "seed": seed,
                                 "name": f"{bl}_{ctx['tag']}_h{cap}_seed{seed}"})
    return cfgs


def complexity_scaling_sweep() -> List[Dict[str, Any]]:
    """Throughput + FLOPs vs total element count.

    Per Ndim, vary spatial-axis size S so that prod(D_i) ranges over ~4 orders
    of magnitude. Time axis fixed at T=64 (per PROTOCOL §5 convention). Spec
    template is the PROTOCOL §5 default; m=(2,...,2).
    Produces the points behind tab_complexity.tex and the headline log-log plot
    that empirically validates Thm 1 (O(prod D_i) linear bound).
    """
    import functools, operator
    cfgs = []
    T = 64
    sweeps = {
        1: [(T,) for _ in [1]],                       # 1D has no spatial dim; one shape only
        2: [(T, S) for S in [4, 8, 16, 32, 64, 128]],
        3: [(T, S, S) for S in [4, 8, 16, 24, 32]],
        4: [(T, S, S, S) for S in [4, 6, 8, 12, 16]],
        5: [(T, S, S, S, S) for S in [4, 6, 8, 10]],
    }
    spec_by_ndim = {1: "lll", 2: "[tv]tv[tv]", 3: "[thw]thw[thw]",
                    4: "[txyz]txyz[txyz]", 5: "[tvxyz]tvxyz[tvxyz]"}
    shape_name_by_ndim = {1: "l", 2: "tv", 3: "thw", 4: "txyz", 5: "tvxyz"}
    for ndim, shapes in sweeps.items():
        for shape in shapes:
            n_elems = functools.reduce(operator.mul, shape, 1)
            for seed in SEEDS:
                cfgs.append(dict(
                    shape=list(shape),
                    shape_name=shape_name_by_ndim[ndim],
                    m=[3] * ndim,
                    spec=spec_by_ndim[ndim],
                    mask_axis=0,
                    model="kun", hidden_dim=64, latent_dim=64, kernel="linear",
                    n_epochs=1,                      # wall-clock + FLOPs only
                    batch_size=8, seed=seed,
                    name=f"kun_nd{ndim}_E{n_elems}_seed{seed}",
                ))
    return cfgs


def spec_3d_ablation_sweep() -> List[Dict[str, Any]]:
    """Twelve spec families on Ndim=3, sine field (T, H, W) = (64, 8, 8).

    Canonical PROTOCOL §5 default shape (matches DEFAULT_CONFIGS[3] /
    exp_3d.yaml / main_results_sweep[ndim=3]). Mask along leading (temporal)
    axis per PROTOCOL §3.1.
    """
    shape = (64, 8, 8)  # PROTOCOL canonical: matches DEFAULT_CONFIGS[3] / exp_3d.yaml / main_results_sweep
    # m[i] = count of shape_name[i] in spec; verified per row below.
    families = [
        ("A_seq_HWL",      "hhhwwwlll",         (3, 3, 3)),
        ("A_seq_WHL",      "wwwhhhlll",         (3, 3, 3)),
        ("A_seq_LWH",      "lllhhhwww",         (3, 3, 3)),
        ("B_full_joint",   "[hwl][hwl][hwl]",   (3, 3, 3)),
        ("C_bookend",      "[hwl]hwl[hwl]",     (3, 3, 3)),
        ("C_book_front",   "[hwl][hwl]hwl",     (3, 3, 3)),
        ("C_book_back",    "hwl[hwl][hwl]",     (3, 3, 3)),
        ("D_pair_HW",      "[hw][hw][hw]lll",   (3, 3, 3)),
        ("D_pair_HL",      "[hl][hl][hl]www",   (3, 3, 3)),
        ("D_pair_WL",      "[wl][wl][wl]hhh",   (3, 3, 3)),
        ("E_roundrobin",   "hwlhwlhwl",         (3, 3, 3)),
        ("F_asym_244",     "[hw]w[wl]l[hwl]l",  (2, 4, 4)),
    ]
    cfgs = []
    for name, spec, m in families:
        for seed in SEEDS:
            cfgs.append(dict(
                shape=list(shape), shape_name="lhw", m=list(m), spec=spec,
                mask_axis=0, model="kun",
                hidden_dim=64, latent_dim=64, kernel="linear",
                n_epochs=30, batch_size=16, lr=1e-3, seed=seed,
                name=f"3d_{name}_seed{seed}",
            ))
    return cfgs


def spec_3d_permutation_sweep() -> List[Dict[str, Any]]:
    """Permutation space of HWL grouping/order on Ndim=3, sine field (64, 8, 8).

    All entries share m=(3,3,3) (each axis appears exactly 3 times across the
    chunk sequence) and Ndim=3 canonical shape (T=64, H=W=8, matches
    DEFAULT_CONFIGS[3] / main_results_sweep), so MSE differences come
    purely from how the 9 axis-applications are partitioned into chunks and
    in what order. Five permutation families:

      A. Single-axis sequential  -- 6 orderings of {H,W,L}
      B. Round-robin             -- 6 cyclic startings of {H,W,L}
      C. Pair-joint              -- 3 pair choices x 3 orderings (front/back/interleaved)
      D. Full joint              -- [hwl][hwl][hwl]
      E. Bookend (joint+seq)     -- joint at mid / front / back

    Total: 6 + 6 + 9 + 1 + 3 = 25 specs. Single-seed by default (smoke);
    widen SEEDS slice below for mean +- std.
    """
    shape = (64, 8, 8)
    families: List[Tuple[str, str]] = [
        # A. single-axis sequential (3! = 6)
        ("A_HWL",      "hhhwwwlll"),
        ("A_HLW",      "hhhlllwww"),
        ("A_WHL",      "wwwhhhlll"),
        ("A_WLH",      "wwwlllhhh"),
        ("A_LHW",      "lllhhhwww"),
        ("A_LWH",      "lllwwwhhh"),
        # B. round-robin (3! = 6 cyclic startings)
        ("B_rr_HWL",   "hwlhwlhwl"),
        ("B_rr_HLW",   "hlwhlwhlw"),
        ("B_rr_WHL",   "whlwhlwhl"),
        ("B_rr_WLH",   "wlhwlhwlh"),
        ("B_rr_LHW",   "lhwlhwlhw"),
        ("B_rr_LWH",   "lwhlwhlwh"),
        # C. pair-joint: 3 pair-choices x 3 orderings each
        ("C_HW_front", "[hw][hw][hw]lll"),
        ("C_HW_back",  "lll[hw][hw][hw]"),
        ("C_HW_inter", "[hw]l[hw]l[hw]l"),
        ("C_HL_front", "[hl][hl][hl]www"),
        ("C_HL_back",  "www[hl][hl][hl]"),
        ("C_HL_inter", "[hl]w[hl]w[hl]w"),
        ("C_WL_front", "[wl][wl][wl]hhh"),
        ("C_WL_back",  "hhh[wl][wl][wl]"),
        ("C_WL_inter", "[wl]h[wl]h[wl]h"),
        # D. full joint
        ("D_full",     "[hwl][hwl][hwl]"),
        # E. bookend (joint + sequential interior)
        ("E_book_mid",   "[hwl]hwl[hwl]"),
        ("E_book_front", "[hwl][hwl]hwl"),
        ("E_book_back",  "hwl[hwl][hwl]"),
    ]
    # 8 frequency-support categories = all 2^3 subsets of {L, H, W}; each
    # declares which axes the sine is allowed to vary along. Round-robin
    # assignment inside sample_sine_params, so n_samples = 8*32 = 256
    # (matches the PROTOCOL default, exactly 32 samples per category).
    # The empty mask "const" produces samples that are constant across the
    # whole field (degenerate baseline -- trivially predictable, useful as a
    # sanity row).
    categories = [
        {"name": "const", "mask": [0, 0, 0]},
        {"name": "L",     "mask": [1, 0, 0]},
        {"name": "H",     "mask": [0, 1, 0]},
        {"name": "W",     "mask": [0, 0, 1]},
        {"name": "LH",    "mask": [1, 1, 0]},
        {"name": "LW",    "mask": [1, 0, 1]},
        {"name": "HW",    "mask": [0, 1, 1]},
        {"name": "LHW",   "mask": [1, 1, 1]},
    ]
    n_samples = 8 * 32  # 256 -- balanced across categories (= PROTOCOL default)
    cfgs = []
    for name, spec in families:
        for seed in SEEDS:  # 5 seeds -> mean +- std (25 specs x 5 = 125 runs)
            cfgs.append(dict(
                shape=list(shape), shape_name="lhw", m=[3, 3, 3], spec=spec,
                mask_axis=0, model="kun",
                hidden_dim=64, latent_dim=64, kernel="linear",
                n_epochs=30, batch_size=16, lr=1e-3, seed=seed,
                n_samples=n_samples, categories=categories,
                name=f"3dperm_{name}_seed{seed}",
            ))
    return cfgs


def cross_dim_ablation_sweep() -> List[Dict[str, Any]]:
    """Knob ablations (spec / m / kernel / depth / skip) on the Ndim=2 sine field.

    Uses the PROTOCOL §5 default 2D shape (64, 8), mask_axis=0 (forecasting).
    """
    shape = (64, 8)
    base = dict(
        shape=list(shape), shape_name="tv",
        m=[3, 3], spec="[tv]tv[tv]", mask_axis=0,
        model="kun", hidden_dim=64, latent_dim=64,
        kernel="linear", depth=1, unet_skip=True,
        n_epochs=30, batch_size=32, lr=1e-3,
    )
    variants = [
        ("default",       {}),
        ("seq_tv",        {"spec": "tttvvv"}),
        ("full_joint",    {"spec": "[tv][tv][tv]"}),
        ("m_2_2_short",   {"m": [2, 2], "spec": "[tv][tv]"}),
        ("kernel_lstm",   {"kernel": "lstm"}),
        ("kernel_transformer", {"kernel": "transformer"}),
        ("depth_1",       {"model": "kun", "depth": 1}),
        ("depth_10",      {"model": "kun_stacked", "depth": 10}),
        ("skip_off",      {"unet_skip": False}),
        ("nd_unet_no_spec", {"model": "nd_unet_no_spec"}),
    ]
    cfgs = []
    for name, override in variants:
        for seed in SEEDS:
            cfg = {**base, **override, "seed": seed, "name": f"abl2_{name}_seed{seed}"}
            cfgs.append(cfg)
    return cfgs


def five_d_scaling_sweep() -> List[Dict[str, Any]]:
    """5D scaling: train_time_s vs prod(D_i).

    Time axis fixed at T=64 (PROTOCOL §5 convention); spatial D varies.
    Verifies linear scaling in prod(D_i) at the largest Ndim we ship.
    """
    cfgs = []
    T = 64
    for D in [4, 6, 8, 10]:
        shape = (T, D, D, D, D)
        for seed in SEEDS:
            cfgs.append(dict(
                shape=list(shape), shape_name="tvxyz",
                m=[3] * 5, spec="[tvxyz]tvxyz[tvxyz]", mask_axis=0,
                model="kun", hidden_dim=32, latent_dim=32, kernel="linear",
                n_epochs=20, batch_size=4, lr=1e-3, n_samples=500, seed=seed,
                name=f"5d_D{D}_seed{seed}",
            ))
    return cfgs


def depth_kernel_scaling_sweep() -> List[Dict[str, Any]]:
    """EXP-04: stack-depth scaling law × kernel ablation, on Ndim=2 default.

    Sweeps depth ∈ {1, 2, 4, 8, 12, 16} × kernel ∈ {linear, lstm, transformer}.
    Uses PROTOCOL §5 2D default shape (64, 8), mask_axis=0.
    """
    cfgs = []
    depths = [1, 2, 4, 8, 12, 16]
    kernels = ["linear", "lstm", "transformer"]
    for depth in depths:
        for kernel in kernels:
            for seed in SEEDS:
                model_name = "kun" if depth == 1 else "kun_stacked"
                cfgs.append(dict(
                    shape=[64, 8], shape_name="tv",
                    m=[3, 3], spec="[tv]tv[tv]", mask_axis=0,
                    model=model_name, depth=depth, kernel=kernel,
                    hidden_dim=64, latent_dim=64, unet_skip=True,
                    n_epochs=30, batch_size=32, lr=1e-3, seed=seed,
                    name=f"dk_d{depth}_{kernel}_seed{seed}",
                ))
    return cfgs


SWEEPS = {
    "main_results":          main_results_sweep,
    "baselines_main":        baselines_main_sweep,
    "complexity_scaling":    complexity_scaling_sweep,
    "spec_3d_ablation":      spec_3d_ablation_sweep,
    "spec_3d_permutation":   spec_3d_permutation_sweep,
    "cross_dim_ablation":    cross_dim_ablation_sweep,
    "five_d_scaling":        five_d_scaling_sweep,
    "depth_kernel_scaling":  depth_kernel_scaling_sweep,
}


# ---------------------------------------------------------------------------
# CLI: emit YAML configs, ready to be consumed by `train.py`
# ---------------------------------------------------------------------------

def _csv_int(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _csv_str(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=list(SWEEPS.keys()) + ["all"])
    ap.add_argument("--dry-run", action="store_true",
                    help="print configs to stdout instead of writing YAML files")

    # main_results sweep parameters (ignored for other sweeps)
    g = ap.add_argument_group("main_results sweep params")
    g.add_argument("--seeds", type=_csv_int, default=None,
                   help=f"comma-separated seeds (default: {','.join(map(str, SEEDS))})")
    g.add_argument("--ndims", type=_csv_int, default=None,
                   help="comma-separated Ndim values (default: 1,2,3,4)")
    g.add_argument("--baselines-1d", type=_csv_str, default=None)
    g.add_argument("--baselines-2d", type=_csv_str, default=None)
    g.add_argument("--baselines-3d", type=_csv_str, default=None)
    g.add_argument("--baselines-4d", type=_csv_str, default=None)
    g.add_argument("--models", type=_csv_str, default=None,
                   help="[baselines_main only] models to run; any subset of "
                        f"{','.join(ALL_MODELS)} (default: all of them)")
    g.add_argument("--dataset", default="synthetic",
                   help="[baselines_main only] data source: synthetic | ett | "
                        "ETTh1 | ETTh2 | ETTm1 | ETTm2 | mnist | "
                        "mnist8 | mnist27 | mnist64 (default: synthetic)")
    g.add_argument("--capacity-grid", type=_csv_int, default=None,
                   help="baseline hidden-dim grid; default: main_results uses "
                        "32,64,128,256,512, baselines_main uses a single 128")
    g.add_argument("--no-kun", action="store_true",
                   help="skip the KUN-v2 reference config")
    g.add_argument("--kun-only", action="store_true",
                   help="drop all baselines; emit only KUN reference cells")
    g.add_argument("--kun-hidden", type=int, default=None)
    g.add_argument("--kun-latent", type=int, default=None)
    g.add_argument("--kun-kernel", default=None,
                   choices=[None, "linear", "lstm", "transformer"])
    g.add_argument("--n-epochs", type=int, default=None,
                   help="training epochs baked into each emitted YAML")
    g.add_argument("--batch-size-le3", type=int, default=None,
                   help="batch size for Ndim<=3")
    g.add_argument("--batch-size-gt3", type=int, default=None,
                   help="batch size for Ndim>3")
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--mnist-frames", type=int, default=None,
                   help="[baselines_main only] Moving MNIST time-axis length; "
                        "loads mnist_<res>x<res>_<frames>frame.npy (default 64). "
                        "The matching .npy must be generated first with "
                        "make_mnist_lowres.py --frames <N>")

    args = ap.parse_args()

    # --- knobs shared by main_results and baselines_main ---
    shared_kwargs: Dict[str, Any] = {}
    if args.seeds is not None:           shared_kwargs["seeds"] = args.seeds
    if args.ndims is not None:           shared_kwargs["ndims"] = args.ndims
    if args.capacity_grid is not None:   shared_kwargs["capacity_grid"] = args.capacity_grid
    if args.kun_hidden is not None:      shared_kwargs["kun_hidden_dim"] = args.kun_hidden
    if args.kun_latent is not None:      shared_kwargs["kun_latent_dim"] = args.kun_latent
    if args.kun_kernel is not None:      shared_kwargs["kun_kernel"] = args.kun_kernel
    if args.n_epochs is not None:        shared_kwargs["n_epochs"] = args.n_epochs
    if args.batch_size_le3 is not None:  shared_kwargs["batch_size_le3"] = args.batch_size_le3
    if args.batch_size_gt3 is not None:  shared_kwargs["batch_size_gt3"] = args.batch_size_gt3
    if args.lr is not None:              shared_kwargs["lr"] = args.lr

    # --- main_results-only knobs: legacy per-Ndim baseline list + KUN toggles ---
    main_results_kwargs = dict(shared_kwargs)
    if args.no_kun:    main_results_kwargs["include_kun"] = False
    if args.kun_only:  main_results_kwargs["kun_only"] = True
    per_ndim_baselines = {
        1: args.baselines_1d, 2: args.baselines_2d,
        3: args.baselines_3d, 4: args.baselines_4d,
    }
    if any(v is not None for v in per_ndim_baselines.values()):
        merged = dict(DEFAULT_BASELINES_BY_NDIM)
        for nd, v in per_ndim_baselines.items():
            if v is not None:
                merged[nd] = v
        main_results_kwargs["baselines_by_ndim"] = merged

    # --- baselines_main-only knobs: unified --models / --dataset selection ---
    baselines_kwargs = dict(shared_kwargs)
    if args.models is not None:        baselines_kwargs["models"] = args.models
    if args.mnist_frames is not None:  baselines_kwargs["mnist_frames"] = args.mnist_frames
    baselines_kwargs["dataset"] = args.dataset

    targets = SWEEPS.keys() if args.sweep == "all" else [args.sweep]
    for name in targets:
        fn = SWEEPS[name]
        if name == "main_results":
            cfgs = fn(**main_results_kwargs)
        elif name == "baselines_main":
            cfgs = fn(**baselines_kwargs)
        else:
            cfgs = fn()
        if args.dry_run:
            print(f"=== {name}: {len(cfgs)} configs ===")
            for cfg in cfgs[:3]:
                print(json.dumps(cfg, indent=2))
            print(f"... ({len(cfgs)} total)")
            continue
        out_dir = CONFIGS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale configs from prior runs so renamed/dropped cells don't linger.
        for stale in out_dir.glob("*.yaml"):
            stale.unlink()
        for cfg in cfgs:
            with open(out_dir / f"{cfg['name']}.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[{name}] wrote {len(cfgs)} configs to {out_dir}")


if __name__ == "__main__":
    main()
