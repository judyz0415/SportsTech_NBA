"""Permuted-label null check for ROCKET + TabPFN.

Labels are randomly permuted so they no longer align with IMU traces, then the
same CV protocol as the real run is applied. **ROC AUC should stay near 0.5**
(chance for a balanced binary problem); large departures suggest label/metric
bugs or gross evaluation errors. This does **not** replace a full leakage audit
(feature leakage, grouping, etc.); it complements correct cross-validation.

Examples::

    # Fast check (stratified 80/20, same split logic as ``goaltend_classify --holdout``)
    python -m goaltend_tabpfn.tabpfn_label_null --split holdout --seed 0

    # Full LOO (slow; matches default ``goaltend_classify``)
    python -m goaltend_tabpfn.tabpfn_label_null --split loo --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Permute class labels vs traces; expect ROC AUC ≈ 0.5.",
    )
    p.add_argument(
        "--split",
        choices=("loo", "holdout"),
        default="holdout",
        help="CV protocol (default holdout — much faster than LOO)",
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for label permutation")
    p.add_argument(
        "--n-perm",
        type=int,
        default=1,
        metavar="N",
        help="Number of independent permutations (seeds seed, seed+1, …); reports mean ± std of ROC AUC",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data root (default: GOALTEND_DATA_DIR / package default)",
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    import torch

    from goaltend_tabpfn.goaltend_classify import (
        DATA_DIR,
        N_GROUPS,
        N_KERNELS,
        evaluate_goaltend,
        evaluate_goaltend_holdout,
        load_goaltend_dataset,
    )

    if args.n_perm < 1:
        raise SystemExit("--n-perm must be >= 1")

    data_dir = DATA_DIR if args.data_dir is None else args.data_dir

    _, y, _ = load_goaltend_dataset(data_dir)
    y = np.asarray(y)

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {dev}")
    print(
        f"Label null: {args.n_perm} permutation(s), base seed={args.seed}, split={args.split}\n"
    )

    aucs: list[float] = []
    accs: list[float] = []

    for k in range(args.n_perm):
        rng = np.random.default_rng(args.seed + k)
        y_perm = rng.permutation(y.copy())

        if args.split == "loo":
            result, _, _ = evaluate_goaltend(
                data_dir=data_dir,
                n_groups=N_GROUPS,
                n_kernels=N_KERNELS,
                device=dev,
                y_override=y_perm,
            )
        else:
            result, *_ = evaluate_goaltend_holdout(
                data_dir=data_dir,
                n_groups=N_GROUPS,
                n_kernels=N_KERNELS,
                device=dev,
                y_override=y_perm,
            )

        aucs.append(float(result["roc_auc"]))
        accs.append(float(result["acc"]))

    print("\n── Summary (permuted labels, traces unchanged) ──")
    if args.n_perm == 1:
        print(f"  ROC AUC  : {aucs[0]:.4f}  (expect ~0.5)")
        print(f"  accuracy : {accs[0]:.4f}")
    else:
        print(
            f"  ROC AUC  : {float(np.mean(aucs)):.4f} ± {float(np.std(aucs)):.4f}  (expect ~0.5)\n"
            f"  accuracy : {float(np.mean(accs)):.4f} ± {float(np.std(accs)):.4f}"
        )


if __name__ == "__main__":
    main(_parse_args())
