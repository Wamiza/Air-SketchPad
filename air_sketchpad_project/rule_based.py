"""Rule-based finger-counting baseline (formulation Sec. 2.6 / 2.7).

A finger counts as extended when its tip is farther from the wrist than its PIP joint.
The pattern of extended fingers is then mapped directly to a gesture.
"""
import numpy as np

from features import FINGERS, K, NUM_LANDMARKS, IDLE, DRAW, PAUSE, COLOUR, ERASE, CLEAR

_PATTERNS = {
    (0, 0, 0, 0): ERASE,   # fist
    (1, 0, 0, 0): DRAW,    # index only
    (1, 1, 0, 0): PAUSE,   # index + middle
    (1, 1, 1, 0): COLOUR,  # three fingers
    (1, 1, 1, 1): CLEAR,   # open palm
}


def fingers_extended(feat):
    """feat: 63-d normalised feature vector (wrist at the origin). Returns a 4-tuple of 0/1."""
    pts = np.asarray(feat, dtype=np.float64).reshape(NUM_LANDMARKS, 3)
    flags = []
    for tip, pip in FINGERS.values():
        flags.append(int(np.linalg.norm(pts[tip]) > np.linalg.norm(pts[pip])))
    return tuple(flags)


def predict_one(feat):
    return _PATTERNS.get(fingers_extended(feat), IDLE)


class RuleBasedClassifier:
    """Minimal scikit-learn-like wrapper so the baseline can be used like any other model."""

    classes_ = np.arange(K)

    def predict(self, X):
        return np.array([predict_one(row) for row in np.atleast_2d(X)])

    def predict_proba(self, X):
        labels = self.predict(X)
        proba = np.zeros((len(labels), K))
        proba[np.arange(len(labels)), labels] = 1.0
        return proba

    def fit(self, X, y=None):  # nothing to learn
        return self
