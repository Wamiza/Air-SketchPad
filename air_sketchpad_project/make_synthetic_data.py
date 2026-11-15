"""DEVELOPMENT ONLY: generate fake hand landmarks so the pipeline can be tested without a webcam.

The data come from a crude geometric hand model. Results on this data say NOTHING about real
performance. Never report them in the assignment; collect real data with collect_data.py.

Usage:
    python make_synthetic_data.py --out data/synthetic_landmarks.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from features import CSV_COLUMNS, GESTURES, K, landmarks_to_features

# (index, middle, ring, pinky) extended? for each gesture class
PATTERNS = {1: (1, 0, 0, 0), 2: (1, 1, 0, 0), 3: (1, 1, 1, 0), 4: (0, 0, 0, 0), 5: (1, 1, 1, 1)}
# Finger geometry in "hand units": MCP offset from wrist and the three segment lengths.
FINGER_GEOM = [((-0.060, -0.28), (0.10, 0.07, 0.06)),
               ((-0.020, -0.30), (0.11, 0.08, 0.06)),
               ((0.020, -0.28), (0.10, 0.07, 0.06)),
               ((0.060, -0.24), (0.08, 0.05, 0.05))]


def _finger_points(mcp, lens, amount, spread, rng):
    """4 points (MCP, PIP, DIP, TIP). amount=1 straight, amount=0 curled into a fist."""
    mcp = np.array(mcp)
    l1, l2, l3 = lens
    up = np.array([np.sin(spread), -np.cos(spread)])
    ext = [mcp, mcp + up * l1, mcp + up * (l1 + l2), mcp + up * (l1 + l2 + l3)]
    pip = mcp + np.array([0, -0.6 * l1])
    dip = pip + np.array([0, 0.7 * l2])
    tip = dip + np.array([0, 0.6 * l3])
    cur = [mcp, pip, dip, tip]
    return [(1 - amount) * np.array(c) + amount * np.array(e) for c, e in zip(cur, ext)]


def hand_landmarks(ext_flags, thumb_out, rng, scale, rot, centre, aspect, noise=0.006):
    """Return (21, 3) MediaPipe-style normalised landmarks for the given finger pattern."""
    pts = [np.array([0.0, 0.0])]
    thumb_dir = np.array([-0.7, -0.7]) if thumb_out else np.array([-0.15, -0.9])
    base = np.array([-0.07, -0.05])
    t = [base + thumb_dir * 0.00, base + thumb_dir * 0.06, base + thumb_dir * 0.12, base + thumb_dir * 0.17]
    pts += [np.array(p) for p in t]
    for (mcp, lens), e in zip(FINGER_GEOM, ext_flags):
        amount = rng.uniform(0.88, 1.0) if e else rng.uniform(0.0, 0.2)
        pts += _finger_points(mcp, lens, amount, rng.normal(0, 0.06), rng)
    # reorder: landmarks 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky (already in that order)
    P = np.array(pts) * scale
    c, s = np.cos(rot), np.sin(rot)
    P = P @ np.array([[c, -s], [s, c]]).T
    z = rng.normal(0, 0.01, size=(21, 1)) * scale
    P = P + rng.normal(0, noise, size=P.shape)
    x = centre[0] + P[:, 0] * aspect  # aspect = H / W so the hand keeps its shape in pixels
    y = centre[1] + P[:, 1]
    return np.hstack([x[:, None], y[:, None], z * aspect])


def idle_pattern(rng):
    """A random finger pattern that is NOT one of the five command patterns."""
    while True:
        pat = tuple(int(v) for v in rng.integers(0, 2, size=4))
        if pat not in PATTERNS.values():
            return pat


def generate(n_participants=6, frames=150, width=640, height=480, seed=0):
    rng = np.random.default_rng(seed)
    aspect = height / width
    rows = []
    for p in range(n_participants):
        scale = rng.uniform(0.9, 1.4)
        base_rot = rng.normal(0, 0.15)
        for label in range(K):
            for _ in range(frames):
                pat = idle_pattern(rng) if label == 0 else PATTERNS[label]
                rot = base_rot + rng.normal(0, 0.25)
                centre = (rng.uniform(0.3, 0.7), rng.uniform(0.55, 0.85))
                raw = hand_landmarks(pat, rng.random() < 0.5, rng, scale * rng.uniform(0.85, 1.15),
                                     rot, centre, aspect)
                raw_flat, feat = landmarks_to_features(raw, False, width, height)
                rows.append([f"S{p + 1:02d}", label, *raw_flat, *feat])
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_landmarks.csv")
    ap.add_argument("--participants", type=int, default=6)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = generate(args.participants, args.frames, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC rows to {args.out} (for pipeline testing only).")


if __name__ == "__main__":
    main()
