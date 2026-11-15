"""Record labelled hand-landmark data for training (formulation Sec. 2.4, Dataset).

Only landmarks are stored, never camera images.

Usage:
    python collect_data.py --participant P01

Keys:  1 Idle  2 Draw  3 Pause  4 Colour  5 Erase  6 Clear   (press once to start recording
       that gesture after a short countdown; it stops by itself at --frames-per-gesture)
       0 / Space  stop recording    q / Esc  save and quit

Tips for good data: move your hand around, change the distance to the camera, rotate it a
little, change the lighting between sessions, and for "Idle" show many different relaxed or
random poses (also thumbs-up, pointing at the screen, hand half-closed, hand leaving the frame).
Collect at least ~5 participants (friends, family, classmates) with every gesture.
"""
import argparse
import os
import time

import cv2
import pandas as pd

from features import CSV_COLUMNS, GESTURES, K, landmarks_to_features
from hand_detector import HandDetector, draw_hand

KEYS = {ord(str(i + 1)): i for i in range(K)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, help="participant id, e.g. P01")
    ap.add_argument("--out", default="data/landmarks.csv")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames-per-gesture", type=int, default=300)
    ap.add_argument("--countdown", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    existing = pd.read_csv(args.out) if os.path.exists(args.out) else None
    counts = [0] * K
    if existing is not None:
        mine = existing[existing["participant"].astype(str) == args.participant]
        for lab, n in mine["label"].value_counts().items():
            counts[int(lab)] = int(n)

    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not open the camera.")
    h, w = frame.shape[:2]
    det = HandDetector()
    rows, session = [], [0] * K
    active, start_at = None, 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    def save():
        if not rows:
            return
        new = pd.DataFrame(rows, columns=CSV_COLUMNS)
        header = not os.path.exists(args.out)
        new.to_csv(args.out, mode="a", header=header, index=False)
        print(f"Saved {len(rows)} rows to {args.out}")
        rows.clear()

    while ok:
        frame = cv2.flip(frame, 1)
        now = time.time()
        res = det.detect(frame)
        if res is not None:
            draw_hand(frame, res[0])

        if active is not None and now >= start_at:
            if res is not None:
                raw_flat, feat = landmarks_to_features(res[0], res[1], w, h)
                rows.append([args.participant, active, *raw_flat, *feat])
                counts[active] += 1
                session[active] += 1
            if session[active] >= args.frames_per_gesture:
                print(f"Done: {GESTURES[active]}")
                active = None
                save()

        cv2.rectangle(frame, (0, 0), (w, 28 + 22 * 3), (30, 30, 30), -1)
        status = "no hand detected" if res is None else "hand OK"
        cv2.putText(frame, f"Participant {args.participant}  |  {status}", (10, 22), font, 0.6, (255, 255, 255), 1)
        line = "  ".join(f"{i + 1}:{GESTURES[i]}={counts[i]}" for i in range(K))
        cv2.putText(frame, line, (10, 46), font, 0.5, (200, 255, 200), 1)
        cv2.putText(frame, "Press 1-6 to record a gesture, 0 to stop, q to quit", (10, 68), font, 0.5, (200, 200, 200), 1)
        if active is not None:
            wait = start_at - now
            msg = (f"Get ready: {GESTURES[active]} in {wait:.1f}s" if wait > 0
                   else f"RECORDING {GESTURES[active]}  {session[active]}/{args.frames_per_gesture}")
            colour = (0, 200, 255) if wait > 0 else (0, 0, 255)
            cv2.putText(frame, msg, (10, h - 20), font, 0.9, colour, 2)
        cv2.imshow("Collect data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in KEYS:
            active, start_at = KEYS[key], time.time() + args.countdown
            session[active] = 0
        elif key in (ord("0"), 32):
            active = None
            save()
        elif key in (ord("q"), 27):
            break
        ok, frame = cap.read()

    save()
    cap.release()
    cv2.destroyAllWindows()
    det.close()


if __name__ == "__main__":
    main()
