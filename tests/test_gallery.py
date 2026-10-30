import numpy as np
import pytest

from checkpoint.face import normalize
from checkpoint.gallery import Gallery


def unit(*values: float) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[: len(values)] = values
    return normalize(vector)


def test_empty_gallery_identifies_nobody():
    gallery = Gallery()
    assert len(gallery) == 0
    assert gallery.identify(unit(1), 0.3) is None


def test_identifies_best_employee_by_best_sample():
    gallery = Gallery()
    gallery.add(1, "Ana", [unit(1, 0), unit(1, 0.2)])
    gallery.add(2, "Bruno", [unit(0, 1)])
    match = gallery.identify(unit(1, 0.1), 0.5)
    assert match is not None and match.name == "Ana"
    assert match.score == pytest.approx(float(unit(1, 0.1) @ unit(1, 0.2)))
    assert [m.name for m in gallery.scores(unit(1, 0.1))] == ["Ana", "Bruno"]


def test_below_threshold_is_unknown():
    gallery = Gallery()
    gallery.add(1, "Ana", [unit(1, 0)])
    assert gallery.identify(unit(1, 1), 0.9) is None  # cosseno ~0.71


def test_wrong_embedding_size_rejected():
    with pytest.raises(ValueError, match="128"):
        Gallery().add(1, "Ana", [np.ones(64)])


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        normalize(np.zeros(128))
