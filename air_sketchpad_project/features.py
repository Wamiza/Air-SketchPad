"""Shared constants and feature extraction.

Implements the input representation from the problem formulation (Sec. 2.2):

    z_j = (p_j - p_0) / s,   s = max_j ||p_j - p_0||,   x = vec(z_0..z_20) in R^63

where p_j are the 21 MediaPipe hand landmarks (converted to pixel-like units so the
aspect ratio of the camera does not distort the hand shape) and p_0 is the wrist.
Left hands are mirrored into a right-hand frame so one classifier handles both hands.
"""
import numpy as np

# Class index = position in this list (formulation Sec. 2.3, class numbers are index + 1).
GESTURES = ["Idle", "Draw", "Pause", "Colour", "Erase", "Clear"]
K = len(GESTURES)
IDLE, DRAW, PAUSE, COLOUR, ERASE, CLEAR = range(K)

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3
WRIST = 0
INDEX_TIP = 8
# (tip, pip) landmark indices for index, middle, ring, pinky
FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}

RAW_COLS = [f"r{i}" for i in range(NUM_FEATURES)]
FEAT_COLS = [f"f{i}" for i in range(NUM_FEATURES)]
CSV_COLUMNS = ["participant", "label"] + RAW_COLS + FEAT_COLS


def canonicalise(raw, is_left):
    """Return a (21, 3) copy of MediaPipe normalised landmarks, mirrored if it is a left hand."""
    pts = np.asarray(raw, dtype=np.float64).reshape(NUM_LANDMARKS, 3).copy()
    if is_left:
        pts[:, 0] = 1.0 - pts[:, 0]
    return pts


def normalise(raw, width, height):
    """Wrist-centred, scale-normalised 63-d feature vector from (21, 3) normalised landmarks."""
    pts = np.asarray(raw, dtype=np.float64).reshape(NUM_LANDMARKS, 3).copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts[:, 2] *= width  # MediaPipe z uses roughly the same scale as x
    pts -= pts[WRIST]
    s = np.max(np.linalg.norm(pts, axis=1))
    if s < 1e-9:
        return np.zeros(NUM_FEATURES)
    return (pts / s).reshape(-1)


def landmarks_to_features(raw, is_left, width, height):
    """Return (raw_flat, feature_vector); both are 63-d and mirror-canonicalised."""
    canon = canonicalise(raw, is_left)
    return canon.reshape(-1), normalise(canon, width, height)
