"""Testes com os modelos reais; pulados se os modelos não tiverem sido baixados."""

from pathlib import Path

import numpy as np
import pytest

from checkpoint.face import EMBEDDING_SIZE, Face, FaceEngine
from checkpoint.models import DETECTOR, RECOGNIZER, ModelError, ensure_model

MODELS_DIR = Path(__file__).parents[1] / "models"


@pytest.fixture(scope="module")
def engine():
    try:
        detector = ensure_model(DETECTOR, MODELS_DIR, download=False)
        recognizer = ensure_model(RECOGNIZER, MODELS_DIR, download=False)
    except ModelError:
        pytest.skip("modelos não baixados (python -m checkpoint modelos)")
    return FaceEngine(detector, recognizer)


def test_blank_frame_has_no_face(engine):
    assert engine.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_detect_handles_changing_frame_size(engine):
    assert engine.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == []
    assert engine.detect(np.zeros((720, 1280, 3), dtype=np.uint8)) == []


def test_embedding_is_normalized(engine):
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    raw = np.array([200, 120, 200, 240,  # caixa
                    260, 200, 340, 200, 300, 250, 270, 300, 330, 300,  # 5 pontos
                    0.99], dtype=np.float32)
    embedding = engine.embed(frame, Face(raw))
    assert embedding.shape == (EMBEDDING_SIZE,)
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-5)
