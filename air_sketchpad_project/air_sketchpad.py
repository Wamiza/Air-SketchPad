"""Hand-Gesture Air Sketchpad: real-time application.

Webcam frame -> hand landmarks -> normalised features -> trained classifier
-> temporal filter -> canvas action.

Usage:
    python air_sketchpad.py                       # uses models/best_model.joblib if it exists
    python air_sketchpad.py --rule-based          # rule-based baseline instead of a trained model
    python air_sketchpad.py --model models/best_model.joblib --camera 0

Keys:  q / Esc quit   c clear canvas   s save PNG   r switch trained model <-> rule-based
"""
import argparse
import os
import time

import cv2
import joblib
import numpy as np

from canvas import Canvas
from engine import EngineConfig, GestureEngine
from features import GESTURES, K, INDEX_TIP, landmarks_to_features
from hand_detector import HandDetector, draw_hand
from rule_based import RuleBasedClassifier

LEGEND = "1 finger: Draw | 2: Pause | 3: Colour | Fist: Erase | Palm (hold 1 s): Clear"


class GestureClassifier:
    """Wraps any model with predict_proba and always returns a length-K probability vector."""

    def __init__(self, model, name):
        self.model, self.name = model, name

    @classmethod
    def rule_based(cls):
        return cls(RuleBasedClassifier(), "Rule-based")

    @classmethod
    def load(cls, path):
        blob = joblib.load(path)
        return cls(blob["model"], blob["name"])

    def probs(self, feat):
        p = self.model.predict_proba(np.asarray(feat).reshape(1, -1))[0]
        out = np.zeros(K)
        out[np.asarray(self.model.classes_, dtype=int)] = p
        return out


class AirSketchpad:
    def __init__(self, classifier, detector, width, height, config=None):
        self.classifier = classifier
        self.alt_classifier = None
        self.detector = detector
        self.w, self.h = width, height
        self.engine = GestureEngine(config)
        self.canvas = Canvas(width, height)
        self.fps = 0.0
        self._last_t = None

    def toggle_classifier(self):
        if self.alt_classifier is not None:
            self.classifier, self.alt_classifier = self.alt_classifier, self.classifier
            self.engine.reset()

    def process(self, frame, t):
        """frame: mirrored BGR image of size (h, w). Returns (display_image, EngineOutput)."""
        if self._last_t is not None and t > self._last_t:
            inst = 1.0 / (t - self._last_t)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last_t = t

        det = self.detector.detect(frame)
        probs, tip, raw = None, None, None
        if det is not None:
            raw, is_left = det
            _, feat = landmarks_to_features(raw, is_left, self.w, self.h)
            probs = self.classifier.probs(feat)
            tip = (float(raw[INDEX_TIP, 0]), float(raw[INDEX_TIP, 1]))  # un-mirrored position

        out = self.engine.update(t, probs, tip)
        self.canvas.apply(out.events)

        display = self.canvas.overlay(frame)
        if raw is not None:
            draw_hand(display, raw)
        self._draw_hud(display, out)
        return display, out

    # -- on-screen display -------------------------------------------------------------------
    def _draw_hud(self, img, out):
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img, (0, 0), (w, 64), (30, 30, 30), -1)
        name = GESTURES[out.stable_label] if out.hand_present else "No hand"
        cv2.putText(img, f"Gesture: {name}", (12, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Model: {self.classifier.name}   FPS: {self.fps:4.1f}", (12, 54),
                    font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        # confidence bar
        cv2.rectangle(img, (w - 220, 14), (w - 20, 30), (80, 80, 80), 1)
        cv2.rectangle(img, (w - 220, 14), (w - 220 + int(200 * out.confidence), 30), (0, 200, 0), -1)
        cv2.putText(img, "confidence", (w - 220, 50), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        # current colour swatch
        cv2.rectangle(img, (w - 300, 14), (w - 270, 44), self.canvas.colour_bgr, -1)
        cv2.rectangle(img, (w - 300, 14), (w - 270, 44), (255, 255, 255), 1)
        # clear-hold progress
        if out.clear_progress > 0:
            bw = int(w * 0.4)
            x0, y0 = (w - bw) // 2, h - 70
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + 14), (80, 80, 80), 1)
            cv2.rectangle(img, (x0, y0), (x0 + int(bw * out.clear_progress), y0 + 14), (0, 0, 255), -1)
            cv2.putText(img, "Hold to clear canvas", (x0, y0 - 6), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cursor marker
        if out.cursor is not None and out.hand_present:
            x0, y0, x1, y1 = self.engine.cfg.active_region
            cx = int((x0 + out.cursor[0] * (x1 - x0)) * w)
            cy = int((y0 + out.cursor[1] * (y1 - y0)) * h)
            cv2.circle(img, (cx, cy), 9, self.canvas.colour_bgr, 2, cv2.LINE_AA)
        cv2.putText(img, LEGEND, (12, h - 12), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def parse_args():
    ap = argparse.ArgumentParser(description="Hand-Gesture Air Sketchpad")
    ap.add_argument("--model", default="models/best_model.joblib", help="trained model file")
    ap.add_argument("--rule-based", action="store_true", help="use the rule-based baseline")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="run on a video file instead of the webcam")
    ap.add_argument("--no-flip", action="store_true", help="do not mirror the input")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--headless", action="store_true", help="no window (for testing)")
    ap.add_argument("--max-frames", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    trained = None
    if not args.rule_based and os.path.exists(args.model):
        trained = GestureClassifier.load(args.model)
        print(f"Loaded model: {trained.name}")
    elif not args.rule_based:
        print(f"No trained model found at {args.model}; using the rule-based baseline. "
              f"Run collect_data.py and train_compare.py first.")
    rule = GestureClassifier.rule_based()

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera / video.")
    h, w = frame.shape[:2]

    app = AirSketchpad(trained or rule, HandDetector(), w, h)
    app.alt_classifier = rule if trained else None
    flip = not (args.no_flip or args.video)
    n, t0 = 0, time.perf_counter()
    while ok:
        if flip:
            frame = cv2.flip(frame, 1)
        t = time.perf_counter() - t0
        display, _ = app.process(frame, t)
        n += 1
        if not args.headless:
            cv2.imshow("Hand-Gesture Air Sketchpad", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                app.canvas.clear()
            elif key == ord("s"):
                path = f"sketch_{int(time.time())}.png"
                cv2.imwrite(path, app.canvas.img)
                print("Saved", path)
            elif key == ord("r"):
                app.toggle_classifier()
        if args.max_frames and n >= args.max_frames:
            break
        ok, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    app.detector.close()


if __name__ == "__main__":
    main()
