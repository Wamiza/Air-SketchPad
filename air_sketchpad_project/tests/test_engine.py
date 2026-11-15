import numpy as np

from engine import EngineConfig, GestureEngine
from features import CLEAR, COLOUR, DRAW, ERASE, IDLE, K, PAUSE


def onehot(label, conf=0.95):
    p = np.full(K, (1 - conf) / (K - 1))
    p[label] = conf
    return p


def run(engine, label, start, seconds, fps=30, conf=0.95, tip=(0.5, 0.5)):
    """Feed `label` for `seconds`; return (all events, last output, end time)."""
    events, out, t = [], None, start
    for _ in range(int(seconds * fps)):
        out = engine.update(t, onehot(label, conf), tip)
        events += out.events
        t += 1.0 / fps
    return events, out, t


def kinds(events):
    return [k for k, _ in events]


def test_draw_starts_after_hold_and_ends_with_stroke_end():
    e = GestureEngine()
    ev, _, t = run(e, DRAW, 0.0, 1.0)
    assert kinds(ev).count("draw") > 15
    ev2, _, _ = run(e, PAUSE, t, 0.5)
    assert kinds(ev2).count("stroke_end") == 1
    # the majority vote keeps "Draw" for at most half a window (3 of 7 frames, ~0.1 s) after the switch
    assert kinds(ev2).count("draw") <= e.cfg.window // 2
    assert kinds(ev2).index("stroke_end") == kinds(ev2).count("draw")


def test_single_frame_flicker_is_suppressed_by_majority_vote():
    e = GestureEngine()
    _, _, t = run(e, DRAW, 0.0, 1.0)
    out = e.update(t, onehot(ERASE), (0.5, 0.5))   # one noisy frame
    assert out.stable_label == DRAW
    assert "erase" not in kinds(out.events)


def test_low_confidence_is_treated_as_idle():
    e = GestureEngine()
    _, out, _ = run(e, DRAW, 0.0, 1.0, conf=0.4)   # below tau = 0.6
    assert out.stable_label == IDLE
    assert out.events == [] or kinds(out.events) == ["stroke_end"]


def test_clear_needs_one_second_hold_and_fires_once():
    e = GestureEngine()
    ev, out, t = run(e, CLEAR, 0.0, 0.8)
    assert "clear" not in kinds(ev)               # held for < 1 s: no deletion
    assert 0 < out.clear_progress < 1
    ev, _, t = run(e, CLEAR, t, 0.6)
    assert kinds(ev).count("clear") == 1          # fires exactly once
    ev, _, _ = run(e, CLEAR, t, 2.0)
    assert "clear" not in kinds(ev)               # no repeat while still held


def test_brief_clear_pose_never_triggers():
    e = GestureEngine()
    _, _, t = run(e, IDLE, 0.0, 0.5)
    ev1, _, t = run(e, CLEAR, t, 0.5)             # passing through an open palm for 0.5 s
    ev2, _, _ = run(e, IDLE, t, 1.0)
    assert "clear" not in kinds(ev1 + ev2)


def test_colour_is_edge_triggered():
    e = GestureEngine()
    ev, _, t = run(e, COLOUR, 0.0, 2.0)
    assert kinds(ev).count("colour") == 1
    _, _, t = run(e, IDLE, t, 0.5)
    ev, _, _ = run(e, COLOUR, t, 1.0)
    assert kinds(ev).count("colour") == 1


def test_erase_events_follow_cursor():
    e = GestureEngine()
    ev, _, _ = run(e, ERASE, 0.0, 0.6)
    assert kinds(ev).count("erase") > 5


def test_cursor_is_smoothed_and_clamped():
    e = GestureEngine(EngineConfig(ema_alpha=0.8))
    e.update(0.0, onehot(PAUSE), (0.5, 0.45))
    out = e.update(0.03, onehot(PAUSE), (0.9, 0.45))
    first = out.cursor[0]
    out2 = e.update(0.06, onehot(PAUSE), (0.9, 0.45))
    assert out2.cursor[0] > first                  # moves gradually, not instantly
    assert 0.0 <= out2.cursor[0] <= 1.0 and 0.0 <= out2.cursor[1] <= 1.0


def test_no_hand_never_draws():
    e = GestureEngine()
    _, _, t = run(e, DRAW, 0.0, 0.5)
    for i in range(30):
        out = e.update(t + i / 30, None, None)
        assert "draw" not in kinds(out.events)
        assert not out.hand_present
