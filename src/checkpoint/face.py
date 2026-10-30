"""Detecção (YuNet) e extração de embedding (SFace) com o OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

EMBEDDING_SIZE = 128


@dataclass(frozen=True)
class Face:
    # Linha crua do YuNet: x, y, w, h, 5 pontos (olhos, nariz, cantos da boca) e score.
    raw: np.ndarray

    @property
    def box(self) -> tuple[int, int, int, int]:
        x, y, w, h = self.raw[:4]
        return int(x), int(y), int(w), int(h)

    @property
    def score(self) -> float:
        return float(self.raw[14])

    @property
    def size(self) -> int:
        """Menor lado da caixa do rosto, em pixels."""
        _, _, w, h = self.box
        return min(w, h)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("embedding nulo")
    return vector / norm


def _quiet_opencv() -> None:
    # O OpenCV 5 avisa a cada modelo carregado que "targets" não são suportados; é inofensivo.
    utils_logging = getattr(getattr(cv2, "utils", None), "logging", None)
    if utils_logging is not None:
        utils_logging.setLogLevel(utils_logging.LOG_LEVEL_ERROR)


class FaceEngine:
    def __init__(self, detector_path: Path, recognizer_path: Path, detect_score: float = 0.8):
        _quiet_opencv()
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), detect_score, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Face]:
        """Rostos no frame, do maior para o menor."""
        h, w = frame.shape[:2]
        if self._input_size != (w, h):
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        found = [Face(row.copy()) for row in faces]
        found.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        return found

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        """Vetor de 128 posições, normalizado (similaridade = produto escalar)."""
        aligned = self._recognizer.alignCrop(frame, face.raw)
        return normalize(self._recognizer.feature(aligned))
