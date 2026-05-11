"""
Random search for AdaBoost hyperparameters maximizing **close-call-only OOF** accuracy
(protocol: all segmented ∪ CC train folds; test folds = CC only).

Run::

    PYTHONPATH=src GOALTEND_LABELS_PATH=data/cleaned_ground_truth.csv \\
      python -m goaltend_close_call.adaboost_opt --trials 160 --top-verify 10
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .close_call_benchmark import CV_N_SPLITS, CV_RANDOM_STATE, _thresh_tuned_acc
from .close_call_cv import stratified_kfold_eval_close_calls
from .close_call_model import build_base_binary, build_usable_close_calls_df
from .paths import outputs_dir


def _sanitize(df_seg, df_cc, feat_cols):
    for df in (df_seg, df_cc):
        raw = df[feat_cols].values.astype(np.float64)
        df[feat_cols] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def _norm_mf(val) -> str | None:
    if val is None:
        return None
    return str(val)


@dataclass
class AdaTrial:
    max_depth: int
    min_samples_leaf: int
    min_samples_split: int
    n_estimators: int
    learning_rate: float
    class_weight_tree: str | None
    max_features_tree: str | None
    calibrate_sigmoid_inner_cv: bool
    oof_acc_05: float
    oof_acc_thr_tuned: float
    best_thresh: float
    seed_fold: int
    verified_mean_acc05: float | None = None
    verified_mean_thr: float | None = None
    verified_acc05_scores: str = ""
    verified_thr_scores: str = ""


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(rng.integers(1, 6)),
        "min_samples_leaf": int(rng.integers(2, 26)),
        "min_samples_split": int(rng.choice([2, 3, 4, 6, 10, 16, 24])),
        "n_estimators": int(rng.integers(65, 360)),
        "learning_rate": float(rng.uniform(0.08, 1.35)),
        "class_weight_tree": None if rng.random() > 0.52 else "balanced",
        "max_features_tree": _norm_mf(rng.choice([None, None, "sqrt", "log2"])),
        # Inner CV calibration is costly; keeps search focused on AdaBoost knobs.
        "calibrate_sigmoid_inner_cv": bool(rng.random() < 0.08),
    }


def trial_to_param_dict(row: AdaTrial) -> dict:
    return {
        "max_depth": row.max_depth,
        "min_samples_leaf": row.min_samples_leaf,
        "min_samples_split": row.min_samples_split,
        "n_estimators": row.n_estimators,
        "learning_rate": row.learning_rate,
        "class_weight_tree": row.class_weight_tree,
        "max_features_tree": row.max_features_tree,
        "calibrate_sigmoid_inner_cv": row.calibrate_sigmoid_inner_cv,
    }


def build_adaboost_pipe(params: dict, *, rnd_state: int) -> Pipeline:
    cw = params["class_weight_tree"]
    est = DecisionTreeClassifier(
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        max_features=params["max_features_tree"],
        class_weight=cw,
        random_state=rnd_state + 101,
    )
    ada_kw = dict(
        estimator=est,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        random_state=rnd_state,
    )

    clf: AdaBoostClassifier | CalibratedClassifierCV
    if params.get("calibrate_sigmoid_inner_cv"):
        ada = AdaBoostClassifier(**ada_kw)
        clf = CalibratedClassifierCV(
            estimator=ada,
            method="sigmoid",
            cv=min(5, max(3, int(os.environ.get("GOALTEND_ADA_CALIB_CV", "4")))),
        )
    else:
        clf = AdaBoostClassifier(**ada_kw)

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def eval_pipe(
    make_pipe,
    df_base,
    df_cc,
    feat_cols,
    *,
    n_splits: int,
    rnd_state: int,
) -> tuple[float, float, float]:
    r = stratified_kfold_eval_close_calls(
        df_base,
        df_cc,
        feat_cols,
        make_pipeline=make_pipe,
        include_segmented=True,
        n_splits=n_splits,
        random_state=rnd_state,
    )
    df_oof = r["oof_predictions_df"]
    y_cc = df_oof["y"].values
    pgoal = np.nan_to_num(
        np.asarray(df_oof["P_goaltend"].values, dtype=np.float64),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    acc05 = float(r["oof_accuracy_close_calls"])
    thr_acc = float(_thresh_tuned_acc(y_cc, pgoal))
    ys = np.array([1 if s == "goaltend" else 0 for s in y_cc], dtype=np.int32)
    best_t, ba = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 513):
        pr = np.where(pgoal >= t, 1, 0).astype(np.int32)
        a = float(np.mean(pr == ys))
        if a > ba:
            ba, best_t = a, float(t)
    return acc05, thr_acc, float(best_t)


def trials_from_samples(p: dict, *, acc05, thr_acc, bt, seed_fold):
    mf = _norm_mf(p.get("max_features_tree"))
    return AdaTrial(
        max_depth=int(p["max_depth"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        min_samples_split=int(p["min_samples_split"]),
        n_estimators=int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        class_weight_tree=p["class_weight_tree"],  # None | "balanced"
        max_features_tree=mf,
        calibrate_sigmoid_inner_cv=bool(p["calibrate_sigmoid_inner_cv"]),
        oof_acc_05=acc05,
        oof_acc_thr_tuned=thr_acc,
        best_thresh=bt,
        seed_fold=int(seed_fold),
    )


def run_random_search(trials: int, rnd_state: int, n_splits: int | None) -> list[AdaTrial]:
    k = CV_N_SPLITS if n_splits is None else int(n_splits)
    df_base, feat_cols = build_base_binary()
    df_cc = build_usable_close_calls_df(feat_cols)
    _sanitize(df_base, df_cc, feat_cols)

    rng = np.random.default_rng(rnd_state + 90210)
    results: list[AdaTrial] = []

    for t in range(trials):
        p = _sample_params(rng)
        ada_seed = int(rnd_state + t * 1337)

        def make_pipe(pi=p, ada=ada_seed):
            return build_adaboost_pipe(pi, rnd_state=ada)

        try:
            # Same CV partition for every trial → hyperparams comparable.
            acc05, thr_acc, bt = eval_pipe(
                make_pipe, df_base, df_cc, feat_cols, n_splits=k, rnd_state=rnd_state
            )
        except Exception as exc:
            print(f"trial {t} skip: {exc}")
            continue

        results.append(trials_from_samples(p, acc05=acc05, thr_acc=thr_acc, bt=bt, seed_fold=ada_seed))
        if trials <= 40 or (t + 1) % max(trials // 10, 1) == 0:
            best_so_far = max(results, key=lambda r: (r.oof_acc_05, r.oof_acc_thr_tuned))
            print(
                f"  [{t+1}/{trials}] best oof@{0.5}: {best_so_far.oof_acc_05:.5f} "
                f"thr_tune: {best_so_far.oof_acc_thr_tuned:.5f}",
                flush=True,
            )

    return results


def multi_seed_verify(
    params: dict,
    *,
    seeds: list[int],
    n_splits: int,
    df_base,
    df_cc,
    feat_cols,
) -> tuple[float, float, list[float], list[float]]:
    accs: list[float] = []
    thrs: list[float] = []
    for s in seeds:
        def mk(p=params, rnd=s):
            return build_adaboost_pipe(p, rnd_state=rnd)

        acc05, thr_acc, _ = eval_pipe(mk, df_base, df_cc, feat_cols, n_splits=n_splits, rnd_state=s)
        accs.append(acc05)
        thrs.append(thr_acc)
    return (
        float(np.mean(accs)),
        float(np.mean(thrs)),
        accs,
        thrs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--top-verify", type=int, default=12)
    ap.add_argument("--verify-seeds", type=int, default=7)
    ap.add_argument("--splits", type=int, default=None)
    args = ap.parse_args()

    n_splits = args.splits or CV_N_SPLITS
    out_dir = outputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    trials_out = run_random_search(args.trials, CV_RANDOM_STATE, n_splits)
    if not trials_out:
        print("No successful trials.")
        return

    trials_out.sort(key=lambda r: (r.oof_acc_05, r.oof_acc_thr_tuned), reverse=True)

    pd.DataFrame([asdict(x) for x in trials_out]).to_csv(
        out_dir / "adaboost_random_search_trials.csv", index=False
    )

    top = trials_out[: args.top_verify]
    df_base, feat_cols = build_base_binary()
    df_cc = build_usable_close_calls_df(feat_cols)
    _sanitize(df_base, df_cc, feat_cols)
    vseeds = [CV_RANDOM_STATE + i * 71 for i in range(args.verify_seeds)]

    for row in top:
        pdict = trial_to_param_dict(row)
        m_acc, m_thr, la, lt = multi_seed_verify(
            pdict,
            seeds=vseeds,
            n_splits=n_splits,
            df_base=df_base,
            df_cc=df_cc,
            feat_cols=feat_cols,
        )
        row.verified_mean_acc05 = m_acc
        row.verified_mean_thr = m_thr
        row.verified_acc05_scores = ";".join(f"{float(x):.5f}" for x in la)
        row.verified_thr_scores = ";".join(f"{float(x):.5f}" for x in lt)

    top.sort(
        key=lambda r: (
            float(r.verified_mean_acc05 or -1),
            float(r.verified_mean_thr or -1),
            r.oof_acc_05,
            r.oof_acc_thr_tuned,
        ),
        reverse=True,
    )
    stable = top[0]
    search_leader = trials_out[0]

    champ_path = out_dir / "adaboost_best_params.json"
    champ_payload = {
        "search_leader_by_oof_accuracy": asdict(search_leader),
        "most_stable_among_top_k_by_mean_oof_seed_sweep": asdict(stable),
    }
    champ_path.write_text(json.dumps(champ_payload, indent=2), encoding="utf-8")

    print("\n=== Search leader (same CV folds for all trials; maximize OOF acc@0.5) ===")
    for k in asdict(search_leader):
        print(f"  {k}: {getattr(search_leader, k)}")
    print("\n=== Stable pick (among top‑k searched, maximize mean acc@0.5 across shuffle seeds) ===")
    for k in asdict(stable):
        print(f"  {k}: {getattr(stable, k)}")
    print("\nSaved:", champ_path)


if __name__ == "__main__":
    main()
