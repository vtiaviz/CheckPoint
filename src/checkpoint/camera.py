"""Abertura da webcam e diagnóstico de captura."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from checkpoint.config import CameraConfig

# Abaixo disso a imagem costuma estar escura demais para o reconhecimento (escala 0 a 255).
MIN_BRIGHTNESS = 70.0
MAX_BRIGHTNESS = 200.0


class CameraError(RuntimeError):
    """A webcam não abriu ou parou de entregar frames."""


@dataclass(frozen=True)
class CameraReport:
    width: int
    height: int
    requested_fps: int
    measured_fps: float
    mean_brightness: float
    frames: int
    backend: str

    @property
    def warnings(self) -> list[str]:
        found = []
        if self.measured_fps < 15:
            found.append(
                f"FPS baixo ({self.measured_fps:.1f}). Feche outros programas que usam a câmera "
                "ou reduza a resolução."
            )
        if self.mean_brightness < MIN_BRIGHTNESS:
            found.append(
                f"Imagem escura (brilho médio {self.mean_brightness:.0f}). "
                "Adicione iluminação frontal no ponto de captura."
            )
        elif self.mean_brightness > MAX_BRIGHTNESS:
            found.append(
                f"Imagem estourada (brilho médio {self.mean_brightness:.0f}). "
                "Evite luz forte apontada para a câmera."
            )
        return found


def mean_brightness(frame: np.ndarray) -> float:
    """Brilho médio do frame em tons de cinza (0 a 255)."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(frame.mean())


def open_camera(cfg: CameraConfig) -> cv2.VideoCapture:
    # No Windows o DirectShow abre a C270 bem mais rápido que o backend padrão (MSMF).
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.index, backend)
    if not cap.isOpened():
        raise CameraError(
            f"não foi possível abrir a câmera {cfg.index}. "
            "Verifique se ela está conectada e se outro programa não a está usando."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    return cap


def diagnose(cfg: CameraConfig, frames: int = 90, preview: bool = False) -> CameraReport:
    """Captura alguns frames e mede resolução real, FPS e brilho."""
    cap = open_camera(cfg)
    try:
        # A C270 ajusta a exposição nos primeiros frames; descartá-los evita brilho falso.
        for _ in range(10):
            cap.read()

        brightness: list[float] = []
        shape: tuple[int, ...] | None = None
        start = time.perf_counter()
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CameraError("a câmera parou de entregar frames durante o teste")
            shape = frame.shape
            brightness.append(mean_brightness(frame))
            if preview:
                cv2.imshow("CheckPoint - teste da camera (Q para sair)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        elapsed = time.perf_counter() - start
        backend = cap.getBackendName()
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()

    assert shape is not None
    captured = len(brightness)
    return CameraReport(
        width=shape[1],
        height=shape[0],
        requested_fps=cfg.fps,
        measured_fps=captured / elapsed if elapsed > 0 else 0.0,
        mean_brightness=float(np.mean(brightness)),
        frames=captured,
        backend=backend,
    )
