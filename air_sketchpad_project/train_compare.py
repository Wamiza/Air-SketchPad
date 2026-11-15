"""Train and compare gesture classifiers (formulation Sec. 2.4, 2.6 and 2.7).

Models:  KNN, SVM (RBF), Random Forest, MLP  (+ the rule-based finger-counting baseline)

Protocol:
  * Data are split BY PARTICIPANT: some participants are held out as a test set that is used
    only once, for the final report.
  * Hyperparameters are tuned on the training participants with GroupKFold cross-validation
    (macro-F1). The mean +/- std of macro-F1 over the folds is the "validation" score.
  * Selection rule: highest validation macro-F1; if other models are within one standard
    deviation of it, the one with the lowest per-frame inference latency is selected.
  * Optional ablation (--ablation): raw landmarks vs normalised features.

Usage:
    python train_compare.py --data data/landmarks.csv
    python train_compare.py --data data/landmarks.csv --test-participants P05 --ablation
    python train_compare.py --data data/synthetic_landmarks.csv --quick      # pipeline test only
"""
import argparse
import os
import tempfile
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEAT_COLS, GESTURES, K, RAW_COLS
from rule_based import RuleBasedClassifier


def make_models(seed, quick):
    """name -> (pipeline, hyperparameter grid)."""
    knn = (Pipeline([("clf", KNeighborsClassifier())]),
           {"clf__n_neighbors": [3, 7, 15] if quick else [3, 5, 7, 11, 15],
            "clf__weights": ["uniform", "distance"]})
    svm = (Pipeline([("scale", StandardScaler()),
                     ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed))]),
           {"clf__C": [1, 10] if quick else [1, 10, 100],
            "clf__gamma": ["scale", 0.01] if quick else ["scale", 0.01, 0.1]})
    rf = (Pipeline([("clf", RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1))]),
          {"clf__n_estimators": [100] if quick else [100, 300],
           "clf__max_depth": [None, 15],
           "clf__min_samples_leaf": [1, 3]})
    mlp = (Pipeline([("scale", StandardScaler()),
                     ("clf", MLPClassifier(activation="relu", solver="adam", early_stopping=True,
                                           max_iter=300, random_state=seed))]),
           {"clf__hidden_layer_sizes": [(128, 64)] if quick else [(128, 64), (64, 32)],
            "clf__alpha": [1e-4, 1e-2] if quick else [1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-3] if quick else [1e-3, 3e-3]})
    return {"KNN": knn, "SVM": svm, "RandomForest": rf, "MLP": mlp}


def split_participants(df, test_arg, seed):
    parts = sorted(df["participant"].astype(str).unique())
    if len(parts) < 3:
        raise SystemExit(f"Need at least 3 participants (found {len(parts)}): 2+ to train/tune, 1+ to test.")
    if test_arg:
        test = [p.strip() for p in test_arg.split(",")]
        missing = [p for p in test if p not in parts]
        if missing:
            raise SystemExit(f"Unknown participant(s): {missing}. Available: {parts}")
    else:
        rng = np.random.default_rng(seed)
        n_test = int(min(max(1, round(0.2 * len(parts))), len(parts) - 2))
        test = sorted(rng.choice(parts, size=n_test, replace=False).tolist())
    train = [p for p in parts if p not in test]
    if len(train) < 2:
        raise SystemExit("Need at least 2 training participants.")
    return train, test


def tune(name, est, grid, X, y, groups, jobs):
    n_splits = min(5, len(np.unique(groups)))
    gs = GridSearchCV(est, grid, scoring="f1_macro", cv=GroupKFold(n_splits=n_splits),
                      n_jobs=jobs, refit=False)
    gs.fit(X, y, groups=groups)
    res = gs.cv_results_
    i = int(np.argmin(res["rank_test_score"]))
    return res["params"][i], float(res["mean_test_score"][i]), float(res["std_test_score"][i]), n_splits


def latency_ms(model, x, n=200):
    x = x.reshape(1, -1)
    for _ in range(10):
        model.predict_proba(x)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict_proba(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def model_size_kb(model):
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    joblib.dump(model, path)
    size = os.path.getsize(path) / 1024
    os.remove(path)
    return size


def evaluate(model, X, y):
    pred = model.predict(X)
    return (f1_score(y, pred, average="macro", zero_division=0), accuracy_score(y, pred),
            confusion_matrix(y, pred, labels=list(range(K))), pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/landmarks.csv")
    ap.add_argument("--test-participants", help="comma-separated ids to hold out, e.g. P05,P06")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--model-out", default="models/best_model.joblib")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="smaller hyperparameter grids")
    ap.add_argument("--ablation", action="store_true", help="also compare raw vs normalised features")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["participant"] = df["participant"].astype(str)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    if "synthetic" in os.path.basename(args.data).lower():
        print("\n*** WARNING: synthetic data. Use this only to test the pipeline, never report it. ***\n")

    train_parts, test_parts = split_participants(df, args.test_participants, args.seed)
    tr, te = df[df["participant"].isin(train_parts)], df[df["participant"].isin(test_parts)]
    Xtr, ytr, gtr = tr[FEAT_COLS].values, tr["label"].values.astype(int), tr["participant"].values
    Xte, yte = te[FEAT_COLS].values, te["label"].values.astype(int)
    print(f"Participants  train/tune: {train_parts}   test (held out): {test_parts}")
    print(f"Frames        train: {len(tr)}   test: {len(te)}")
    print("Class counts (train):", dict(zip(GESTURES, np.bincount(ytr, minlength=K))))

    rows, fitted, confs, tuned = [], {}, {}, {}
    for name, (est, grid) in make_models(args.seed, args.quick).items():
        print(f"\n[{name}] tuning with GroupKFold ...")
        params, cv_mean, cv_std, n_splits = tune(name, est, grid, Xtr, ytr, gtr, args.jobs)
        print(f"  best params: {params}\n  validation macro-F1 = {cv_mean:.4f} +/- {cv_std:.4f} ({n_splits} folds)")
        final = clone(est).set_params(**params)
        if name == "SVM":
            final.set_params(clf__probability=True)  # needed for confidence gating in the app
        t0 = time.perf_counter()
        final.fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
        if name == "RandomForest":
            final.set_params(clf__n_jobs=1)  # single-frame latency is measured on one core
        f1, acc, cm, _ = evaluate(final, Xte, yte)
        lat = latency_ms(final, Xte[0])
        rows.append(dict(model=name, val_macro_f1_mean=cv_mean, val_macro_f1_std=cv_std,
                         test_macro_f1=f1, test_accuracy=acc, train_time_s=train_s,
                         latency_ms=lat, size_kb=model_size_kb(final), best_params=str(params)))
        fitted[name], confs[name], tuned[name] = final, cm, params
        print(f"  TEST macro-F1 = {f1:.4f}  accuracy = {acc:.4f}  latency = {lat:.2f} ms")

    # rule-based baseline (nothing to train, nothing to tune)
    rb = RuleBasedClassifier()
    f1, acc, cm, _ = evaluate(rb, Xte, yte)
    rows.append(dict(model="Rule-based", val_macro_f1_mean=np.nan, val_macro_f1_std=np.nan,
                     test_macro_f1=f1, test_accuracy=acc, train_time_s=0.0,
                     latency_ms=latency_ms(rb, Xte[0]), size_kb=0.0, best_params="-"))
    confs["Rule-based"] = cm
    print(f"\n[Rule-based baseline] TEST macro-F1 = {f1:.4f}  accuracy = {acc:.4f}")

    table = pd.DataFrame(rows)
    # selection rule (Sec. 2.7)
    ml = table[table["model"] != "Rule-based"]
    best = ml.loc[ml["val_macro_f1_mean"].idxmax()]
    close = ml[ml["val_macro_f1_mean"] >= best["val_macro_f1_mean"] - best["val_macro_f1_std"]]
    chosen = close.loc[close["latency_ms"].idxmin()]["model"]
    print(f"\nHighest validation macro-F1: {best['model']}. Within one std of it: {list(close['model'])}.")
    print(f"SELECTED (fastest among those): {chosen}")

    joblib.dump({"model": fitted[chosen], "name": chosen, "gestures": GESTURES,
                 "params": tuned[chosen], "train_participants": train_parts}, args.model_out)
    print(f"Saved selected model to {args.model_out}")

    # ablation: raw landmarks vs normalised features, same model type and hyperparameters
    abl_text = ""
    if args.ablation:
        est = make_models(args.seed, args.quick)[chosen][0]
        m = clone(est).set_params(**tuned[chosen]).fit(tr[RAW_COLS].values, ytr)
        raw_f1 = f1_score(yte, m.predict(te[RAW_COLS].values), average="macro", zero_division=0)
        norm_f1 = float(table.loc[table["model"] == chosen, "test_macro_f1"].iloc[0])
        abl_text = (f"Ablation ({chosen}): test macro-F1 raw landmarks = {raw_f1:.4f}, "
                    f"normalised features = {norm_f1:.4f}")
        print("\n" + abl_text)

    # outputs
    table.to_csv(os.path.join(args.results_dir, "comparison.csv"), index=False)
    names = list(confs.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2))
    for ax, n in zip(np.atleast_1d(axes), names):
        cm = confs[n].astype(float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(n)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(GESTURES, rotation=60, fontsize=7); ax.set_yticklabels(GESTURES, fontsize=7)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(os.path.join(args.results_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table)); wd = 0.38
    ax.bar(x - wd / 2, table["val_macro_f1_mean"].fillna(0), wd, yerr=table["val_macro_f1_std"].fillna(0),
           label="validation (GroupKFold, mean +/- std)", capsize=3)
    ax.bar(x + wd / 2, table["test_macro_f1"], wd, label="held-out participants")
    ax.set_xticks(x); ax.set_xticklabels(table["model"]); ax.set_ylim(0, 1.05); ax.set_ylabel("macro-F1")
    ax.legend(fontsize=8, loc="lower right"); ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(os.path.join(args.results_dir, "model_comparison.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    with open(os.path.join(args.results_dir, "report.txt"), "w") as f:
        f.write(f"Data: {args.data}\nTrain participants: {train_parts}\nTest participants: {test_parts}\n\n")
        f.write(table.drop(columns=["best_params"]).round(4).to_string(index=False) + "\n\n")
        f.write("Best hyperparameters:\n" + "\n".join(f"  {n}: {p}" for n, p in tuned.items()) + "\n\n")
        f.write(f"Selected model: {chosen}\n{abl_text}\n\n")
        pred = fitted[chosen].predict(Xte)
        f.write(f"Per-class report for {chosen} on held-out participants:\n")
        f.write(classification_report(yte, pred, labels=list(range(K)), target_names=GESTURES, zero_division=0))
    print(f"\nResults written to {args.results_dir}/ (comparison.csv, report.txt, *.png)")
    print(table.drop(columns=["best_params"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
