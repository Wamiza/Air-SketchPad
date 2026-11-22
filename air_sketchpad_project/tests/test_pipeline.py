"""End-to-end tests of the whole application loop using a fake detector that returns synthetic
hands (no webcam needed)."""
import numpy as np

from air_sketchpad import AirSketchpad, GestureClassifier
from features import CLEAR, DRAW, ERASE, PAUSE
from make_synthetic_data import PATTERNS, hand_landmarks

W, H = 640, 480


class FakeDetector:
    def __init__(self):
        self.label, self.centre, self.rng = None, (0.5, 0.6), np.random.default_rng(1)

    def detect(self, frame):
        if self.label is None:
            return None
        raw = hand_landmarks(PATTERNS[self.label], False, self.rng, 1.0, 0.05, self.centre, H / W,
                             noise=0.002)
        return raw, False

    def close(self):
        pass


def make_app():
    det = FakeDetector()
    return AirSketchpad(GestureClassifier.rule_based(), det, W, H), det


def feed(app, det, label, seconds, t, fps=30, move=None):
    frame = np.zeros((H, W, 3), np.uint8)
    for i in range(int(seconds * fps)):
        det.label = label
        if move:
            det.centre = move(i)
        app.process(frame, t)
        t += 1 / fps
    return t


def test_drawing_leaves_ink_and_returns_correct_frame_shape():
    app, det = make_app()
    frame = np.zeros((H, W, 3), np.uint8)
    det.label = DRAW
    t = 0.0
    for i in range(40):
        det.centre = (0.4 + i * 0.005, 0.6)
        display, _ = app.process(frame, t)
        t += 1 / 30
    assert display.shape == (H, W, 3)
    assert app.canvas.has_ink()


def test_pause_moves_cursor_without_drawing():
    app, det = make_app()
    feed(app, det, PAUSE, 1.0, 0.0, move=lambda i: (0.4 + i * 0.004, 0.6))
    assert not app.canvas.has_ink()


def test_erase_gesture_sends_erase_events_to_canvas():
    app, det = make_app()
    seen = []
    original = app.canvas.apply
    app.canvas.apply = lambda events: (seen.extend(k for k, _ in events), original(events))[1]
    feed(app, det, ERASE, 1.0, 0.0, move=lambda i: (0.5, 0.6))
    assert seen.count("erase") > 5
    assert "draw" not in seen


def test_clear_gesture_held_one_second_wipes_canvas_but_not_before():
    app, det = make_app()
    t = feed(app, det, DRAW, 1.0, 0.0, move=lambda i: (0.4 + i * 0.006, 0.6))
    assert app.canvas.has_ink()
    t = feed(app, det, CLEAR, 0.6, t)
    assert app.canvas.has_ink()          # not held long enough yet
    feed(app, det, CLEAR, 1.0, t)
    assert not app.canvas.has_ink()


def test_no_hand_frames_do_not_crash():
    app, det = make_app()
    feed(app, det, None, 0.5, 0.0)


def test_fast_moving_finger_draws_continuous_stroke():
    app, det = make_app()
    # Rapid movement across the frame (speed ~ 0.9 canvas widths/sec)
    feed(app, det, DRAW, 0.8, 0.0, move=lambda i: (0.2 + i * 0.025, 0.5))
    assert app.canvas.has_ink()
    # Check that ink is drawn across the width of the canvas
    assert app.canvas.img[:, :, 2].max() > 0  # Red ink exists
    active_cols = (app.canvas.img[:, :, 2] > 0).any(axis=0).nonzero()[0]
    # Stroke covers a wide span due to fast movement
    assert active_cols.max() - active_cols.min() > W * 0.4
