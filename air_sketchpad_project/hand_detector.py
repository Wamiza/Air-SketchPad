"""Thin wrapper around MediaPipe Hands (Stage 1 of the pipeline, formulation Sec. 2.1).

Backends, tried in this order:
  1. Classic `mediapipe.solutions.hands` (mediapipe <= 0.10.2x). Model is bundled, no download.
  2. MediaPipe Tasks `HandLandmarker` (newer mediapipe). Needs `hand_landmarker.task`, which is
     downloaded once into models/ on first use.

`detect()` expects a BGR frame that is ALREADY mirrored (selfie view). It returns
(raw, is_left) where raw is a (21, 3) array of normalised landmarks, or None if no hand is found.
"""
import os
import urllib.request

import cv2
import numpy as np

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]
TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
            "float16/1/hand_landmarker.task")
TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")


class HandDetector:
    def __init__(self, min_detection=0.4, min_tracking=0.35, fallback_static=True):
        import mediapipe as mp
        self._mp = mp
        self._t0 = None
        self.fallback_static = fallback_static
        self._static_hands = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=0,
                min_detection_confidence=min_detection, min_tracking_confidence=min_tracking)
            if self.fallback_static:
                # Fast motion recovery fallback: runs full-frame palm detection if tracking drops
                self._static_hands = mp.solutions.hands.Hands(
                    static_image_mode=True, max_num_hands=1, model_complexity=0,
                    min_detection_confidence=min_detection)
        else:
            self.backend = "tasks"
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            if not os.path.exists(TASK_PATH):
                print(f"Downloading hand model to {TASK_PATH} ...")
                urllib.request.urlretrieve(TASK_URL, TASK_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=min_detection,
                min_hand_presence_confidence=0.35, min_tracking_confidence=min_tracking)
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._ts = 0

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            res = self._hands.process(rgb)
            if (not res or not res.multi_hand_landmarks) and self._static_hands is not None:
                # If tracking lost the fast-moving hand, immediately recover it via full-frame detector
                res = self._static_hands.process(rgb)
            if not res or not res.multi_hand_landmarks:
                return None
            lm = res.multi_hand_landmarks[0].landmark
            raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
            is_left = res.multi_handedness[0].classification[0].label == "Left"
            return raw, is_left
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33  # monotonically increasing timestamp in ms
        res = self._landmarker.detect_for_video(mp_image, self._ts)
        if not res.hand_landmarks:
            return None
        raw = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float64)
        is_left = res.handedness[0][0].category_name == "Left"
        return raw, is_left

    def close(self):
        if self.backend == "solutions":
            self._hands.close()
            if self._static_hands is not None:
                self._static_hands.close()
        else:
            self._landmarker.close()


def draw_hand(frame, raw, colour=(0, 255, 120)):
    """Draw the 21 landmarks and their connections onto `frame` in place."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in raw]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
