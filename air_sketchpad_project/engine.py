"""Temporal decoding and cursor control (formulation Sec. 2.5).

Turns noisy per-frame class probabilities into stable commands:

  1. Predictions with confidence below `tau` are treated as Idle.
  2. The gesture is the majority vote over the last `window` frames.
  3. A command fires only after that gesture has been stable for `hold_s[gesture]` seconds
     (about one second for the destructive Clear gesture).
  4. The cursor follows the index fingertip: c_t = alpha * c_{t-1} + (1 - alpha) * phi(p_8).

The engine is pure Python/NumPy (no OpenCV), so it is easy to unit-test.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from features import IDLE, DRAW, PAUSE, COLOUR, ERASE, CLEAR


@dataclass
class EngineConfig:
    window: int = 7            # w: frames in the majority vote
    tau: float = 0.60          # confidence threshold, below this -> Idle
    ema_alpha: float = 0.60    # alpha: cursor smoothing (higher = smoother but laggier)
    hold_s: dict = field(default_factory=lambda: {
        IDLE: 0.0, DRAW: 0.08, PAUSE: 0.08, COLOUR: 0.35, ERASE: 0.12, CLEAR: 1.0})
    # phi: part of the camera image (x0, y0, x1, y1 in [0, 1]) that maps to the whole canvas
    active_region: Tuple[float, float, float, float] = (0.15, 0.10, 0.85, 0.80)
    hand_lost_reset_s: float = 0.5  # reset cursor smoothing after the hand was gone this long


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
                self._cursor = (a * self._cursor[0] + (1 - a) * c[0],
                                a * self._cursor[1] + (1 - a) * c[1])
            self._last_hand_t = t

        events = []
        ready = held >= cfg.hold_s[g]
        if hand and ready and self._cursor is not None:
            if g == DRAW:
                events.append(("draw", self._cursor))
                self._drawing = True
            elif g == ERASE:
                events.append(("erase", self._cursor))
        if g != DRAW and self._drawing:
            events.append(("stroke_end", None))
            self._drawing = False
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
