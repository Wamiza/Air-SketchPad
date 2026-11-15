"""Virtual drawing canvas that applies the events produced by GestureEngine."""
import cv2
import numpy as np


class Canvas:
    COLOURS = [("Red", (0, 0, 255)), ("Green", (0, 200, 0)), ("Blue", (255, 80, 0)),
               ("Yellow", (0, 220, 255)), ("White", (255, 255, 255))]

    def __init__(self, width, height, brush=6, eraser=28):
        self.w, self.h = int(width), int(height)
        self.brush, self.eraser = brush, eraser
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.colour_idx = 0
        self._prev = None

    @property
    def colour_name(self):
        return self.COLOURS[self.colour_idx][0]

    @property
    def colour_bgr(self):
        return self.COLOURS[self.colour_idx][1]

    def _px(self, c):
        return (int(round(c[0] * (self.w - 1))), int(round(c[1] * (self.h - 1))))

    def clear(self):
        self.img[:] = 0
        self._prev = None

    def has_ink(self):
        return bool(self.img.any())

    def apply(self, events):
        for kind, arg in events:
            if kind == "draw":
                p = self._px(arg)
                if self._prev is None:
                    cv2.circle(self.img, p, self.brush // 2, self.colour_bgr, -1, cv2.LINE_AA)
                else:
                    cv2.line(self.img, self._prev, p, self.colour_bgr, self.brush, cv2.LINE_AA)
                self._prev = p
            elif kind == "erase":
                cv2.circle(self.img, self._px(arg), self.eraser, (0, 0, 0), -1)
                self._prev = None
            elif kind == "stroke_end":
                self._prev = None
            elif kind == "colour":
                self.colour_idx = (self.colour_idx + 1) % len(self.COLOURS)
            elif kind == "clear":
                self.clear()

    def overlay(self, frame):
        """Return a copy of `frame` (same size as the canvas) with the strokes painted on top."""
        out = frame.copy()
        mask = self.img.any(axis=2)
        out[mask] = self.img[mask]
        return out
