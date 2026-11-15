import numpy as np

from features import canonicalise, landmarks_to_features, normalise
from make_synthetic_data import hand_landmarks


def _hand(seed=0):
    rng = np.random.default_rng(seed)
    return hand_landmarks((1, 0, 0, 0), False, rng, 1.0, 0.1, (0.5, 0.7), 480 / 640)


def test_feature_shape_and_wrist_at_origin():
    f = normalise(_hand(), 640, 480)
    assert f.shape == (63,)
    assert np.allclose(f[:3], 0)
    assert np.isclose(np.max(np.linalg.norm(f.reshape(21, 3), axis=1)), 1.0)


def test_translation_invariance():
    raw = _hand()
    moved = raw.copy()
    moved[:, 0] += 0.1
    moved[:, 1] -= 0.05
    assert np.allclose(normalise(raw, 640, 480), normalise(moved, 640, 480), atol=1e-9)


def test_scale_invariance():
    raw = _hand()
    c = raw[0]
    big = raw.copy()
    big[:, :2] = c[:2] + (raw[:, :2] - c[:2]) * 1.7
    big[:, 2] = raw[:, 2] * 1.7
    assert np.allclose(normalise(raw, 640, 480), normalise(big, 640, 480), atol=1e-9)


def test_left_hand_is_mirrored_to_right_hand_frame():
    raw = _hand()
    mirrored = raw.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]            # what a left hand looks like
    _, f_right = landmarks_to_features(raw, False, 640, 480)
    _, f_left = landmarks_to_features(mirrored, True, 640, 480)
    assert np.allclose(f_right, f_left, atol=1e-9)


def test_degenerate_input_returns_zeros():
    assert not normalise(np.zeros((21, 3)), 640, 480).any()


def test_canonicalise_does_not_modify_input():
    raw = _hand()
    before = raw.copy()
    canonicalise(raw, True)
    assert np.array_equal(raw, before)
