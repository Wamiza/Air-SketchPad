"""Temporal decoding and cursor control (formulation Sec. 2.5).

Turns noisy per-frame class probabilities into stable commands:

  1. Predictions with confidence below `tau` are treated as Idle.
  2. The gesture is the majority vote over the last `window` frames.
  3. A command fires only after that gesture has been stable for `hold_s[gesture]` seconds
     (about one second for the destructive Clear gesture).
  4. The cursor follows the index fingertip: c_t = a_t * c_{t-1} + (1 - a_t) * phi(p_8).
     a_t = alpha when the finger is still, and shrinks when it moves fast (speed-adaptive EMA),
     so a steady hand is smooth but quick strokes keep their sharp corners (V, W, N, ...).
     Set speed_ref=None for the plain fixed-alpha EMA.
  5. Pen protection while writing: a brief misread of the hand pose does not lift the pen
     (`draw_grace_s`), and Erase is blocked for `erase_block_s` after the last drawn point, so
     a finger that bends at the bottom of a stroke cannot rub out what was just written.

The engine is pure Python/NumPy (no OpenCV), so it is easy to unit-test.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math

import numpy as np

from features import IDLE, DRAW, PAUSE, COLOUR, ERASE, CLEAR


@dataclass
class EngineConfig:
    window: int = 7            # w: frames in the majority vote
    tau: float = 0.50          # confidence threshold, below this -> Idle (forgiving to motion blur)
    ema_alpha: float = 0.60    # alpha: cursor smoothing (higher = smoother but laggier)
    min_alpha: float = 0.15    # minimum smoothing factor for fast movements to ensure stability
    hold_s: dict = field(default_factory=lambda: {
        IDLE: 0.0, DRAW: 0.08, PAUSE: 0.08, COLOUR: 0.35, ERASE: 0.30, CLEAR: 1.0})
    # phi: part of the camera image (x0, y0, x1, y1 in [0, 1]) that maps to the whole canvas
    active_region: Tuple[float, float, float, float] = (0.10, 0.08, 0.90, 0.92)
    hand_lost_reset_s: float = 0.5  # reset cursor smoothing after the hand was gone this long
    speed_ref: Optional[float] = 0.4   # smoothing fades out over this much extra speed (None = fixed EMA)
    speed_deadband: float = 0.25       # canvas-widths/s: slower than this counts as jitter and stays fully smoothed
    draw_grace_s: float = 0.30      # a gesture dropout shorter than this does not lift the pen
    erase_block_s: float = 0.60     # no erasing for this long after the last drawn point


@dataclass
class EngineOutput:
    raw_label: int                       # per-frame prediction after confidence gating
    confidence: float
    stable_label: int                    # majority-vote gesture g_t
    held_s: float                        # how long g_t has been stable
    cursor: Optional[Tuple[float, float]]  # smoothed cursor in canvas coords [0, 1]^2
    hand_present: bool
    events: List[tuple] = field(default_factory=list)
    clear_progress: float = 0.0          # 0..1 while the Clear gesture is being held


class GestureEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.cfg = config or EngineConfig()
        self.reset()

    def reset(self):
        self._window = deque(maxlen=self.cfg.window)
        self._g = IDLE
        self._since = 0.0
        self._fired = False
        self._drawing = False
        self._cursor = None
        self._last_hand_t = None
        self._left_draw_t = None
        self._last_draw_t = -1e9

    # -- helpers -------------------------------------------------------------------------
    def _vote(self):
        counts = Counter(self._window)
        best = max(counts.values())
        for label in reversed(self._window):  # ties -> most recent label wins
            if counts[label] == best:
                return label
        return IDLE

    def _phi(self, xy):
        x0, y0, x1, y1 = self.cfg.active_region
        cx = (xy[0] - x0) / (x1 - x0)
        cy = (xy[1] - y0) / (y1 - y0)
        return (float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0)))

    # -- main entry point ------------------------------------------------------------------
    def update(self, t, probs, fingertip) -> EngineOutput:
        """t: time in seconds; probs: (K,) class probabilities or None if no hand;
        fingertip: index fingertip (x, y) in [0, 1] image coordinates or None."""
        cfg = self.cfg
        hand = probs is not None
        if hand:
            k = int(np.argmax(probs))
            conf = float(probs[k])
            raw = k if conf >= cfg.tau else IDLE
        else:
            raw, conf = IDLE, 0.0

        self._window.append(raw)
        g = self._vote()
        if g != self._g:
            self._g, self._since, self._fired = g, t, False
        held = t - self._since

        if hand and fingertip is not None:
            c = self._phi(fingertip)
            stale = self._last_hand_t is None or (t - self._last_hand_t) > cfg.hand_lost_reset_s
            if self._cursor is None or stale:
                self._cursor = c
            else:
                a = cfg.ema_alpha
                if cfg.speed_ref:
                    dt = max(t - self._last_hand_t, 1e-3)
                    speed = math.hypot(c[0] - self._cursor[0], c[1] - self._cursor[1]) / dt
                    a = max(cfg.min_alpha, a * math.exp(-max(0.0, speed - cfg.speed_deadband) / cfg.speed_ref))
                self._cursor = (a * self._cursor[0] + (1 - a) * c[0],
                                a * self._cursor[1] + (1 - a) * c[1])
            self._last_hand_t = t

        events = []
        ready = held >= cfg.hold_s[g]
        if g == ERASE and (t - self._last_draw_t) < cfg.erase_block_s:
            ready = False  # just drew something: a bent finger must not erase it
        if hand and ready and self._cursor is not None:
            if g == DRAW:
                events.append(("draw", self._cursor))
                self._drawing = True
                self._last_draw_t = t
            elif g == ERASE:
                events.append(("erase", self._cursor))
        # lift the pen only if Draw has really been left for longer than the grace period
        if g == DRAW:
            self._left_draw_t = None
        elif self._drawing:
            if self._left_draw_t is None:
                self._left_draw_t = t
            if t - self._left_draw_t >= cfg.draw_grace_s:
                events.append(("stroke_end", None))
                self._drawing = False
                self._left_draw_t = None
        if ready and not self._fired:
            if g == COLOUR:
                events.append(("colour", None))
                self._fired = True
            elif g == CLEAR:
                events.append(("clear", None))
                self._fired = True

        progress = 0.0
        if g == CLEAR:
            progress = 1.0 if self._fired else min(held / cfg.hold_s[CLEAR], 1.0)

        return EngineOutput(raw_label=raw, confidence=conf, stable_label=g, held_s=held,
                            cursor=self._cursor, hand_present=hand, events=events,
                            clear_progress=progress)
