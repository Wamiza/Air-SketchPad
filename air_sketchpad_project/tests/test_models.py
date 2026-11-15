"""Sanity checks on the baseline and a quick end-to-end run of the training script on synthetic
data. (Scores on synthetic data say nothing about real-world accuracy.)"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from features import FEAT_COLS
from make_synthetic_data import generate
from rule_based import RuleBasedClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_rule_based_works_on_clean_synthetic_hands():
    df = generate(n_participants=3, frames=60, seed=3)
    pred = RuleBasedClassifier().predict(df[FEAT_COLS].values)
    assert (pred == df["label"].values).mean() > 0.9


def test_train_compare_runs_end_to_end(tmp_path):
    data = tmp_path / "synthetic_small.csv"
    generate(n_participants=4, frames=40, seed=5).to_csv(data, index=False)
    res = tmp_path / "results"
    model = tmp_path / "best.joblib"
    cmd = [sys.executable, "train_compare.py", "--data", str(data), "--results-dir", str(res),
           "--model-out", str(model), "--quick", "--jobs", "1", "--ablation"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    table = pd.read_csv(res / "comparison.csv")
    assert set(table["model"]) == {"KNN", "SVM", "RandomForest", "MLP", "Rule-based"}
    assert (res / "confusion_matrices.png").exists() and model.exists()
    import joblib
    blob = joblib.load(model)
    proba = blob["model"].predict_proba(np.zeros((1, 63)) + 0.01)
    assert proba.shape[0] == 1 and np.isclose(proba.sum(), 1.0)
