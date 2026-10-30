"""Prova de vida: desafio ativo (MediaPipe Face Landmarker) e anti-spoofing passivo (MiniFASNet).

O desafio ativo barra foto parada: a pessoa precisa piscar ou virar o rosto para o lado
sorteado. O anti-spoofing passivo analisa a textura da imagem e barra foto impressa e tela de
celular, inclusive um vídeo da pessoa fazendo o gesto pedido.
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from checkpoint.face import Face

# Sinais do Face Landmarker (blendshapes vão de 0 a 1).
EYES_OPEN = 0.30
EYES_CLOSED = 0.55
FRONTAL_YAW = 10.0  # graus
TURN_YAW = 20.0  # graus


@dataclass(frozen=True)
class Signals:
    blink: float  # 0 = olhos abertos, 1 = fechados (o maior dos dois olhos)
    yaw: float  # graus; positivo = rosto virado para a esquerda da própria pessoa


def yaw_from_matrix(matrix: np.ndarray) -> float:
    """Ângulo horizontal da cabeça a partir da matriz de transformação facial do MediaPipe.

    Positivo quando a pessoa vira o rosto para a própria esquerda (o nariz vai para a direita
    da imagem não espelhada).
    """
    return math.degrees(math.atan2(float(matrix[0, 2]), float(matrix[2, 2])))


class LandmarkSensor:
    """Extrai piscada e rotação da cabeça de cada frame com o MediaPipe Face Landmarker."""

    def __init__(self, model_path: Path):
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def close(self) -> None:
        self._landmarker.close()

    def measure(self, frame: np.ndarray, timestamp_ms: int) -> Signals | None:
        import mediapipe as mp

        # O modo VIDEO exige timestamps estritamente crescentes.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        if not result.face_blendshapes or not result.facial_transformation_matrixes:
            return None
        shapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        blink = max(shapes.get("eyeBlinkLeft", 0.0), shapes.get("eyeBlinkRight", 0.0))
        yaw = yaw_from_matrix(np.asarray(result.facial_transformation_matrixes[0]))
        return Signals(blink=blink, yaw=yaw)


class Challenge(enum.Enum):
    BLINK = "Pisque os olhos"
    TURN_LEFT = "Vire o rosto para a sua esquerda"
    TURN_RIGHT = "Vire o rosto para a sua direita"

    @property
    def screen_text(self) -> str:
        """Texto sem acentos para a tela do OpenCV, com seta no sentido do espelho."""
        return {
            Challenge.BLINK: "Pisque os olhos",
            Challenge.TURN_LEFT: "<< Vire o rosto para a sua esquerda",
            Challenge.TURN_RIGHT: "Vire o rosto para a sua direita >>",
        }[self]


class Status(enum.Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class ChallengeTracker:
    """Acompanha um desafio frame a frame.

    Piscar: olhos abertos, depois fechados, depois abertos de novo.
    Virar: rosto de frente primeiro, depois virado para o lado pedido. Virar para o lado
    errado reprova, para que um vídeo com vários gestos não passe por tentativa e erro.
    """

    challenge: Challenge
    status: Status = Status.PENDING
    reason: str = ""
    _stage: int = 0

    def update(self, signals: Signals) -> Status:
        if self.status is not Status.PENDING:
            return self.status
        if self.challenge is Challenge.BLINK:
            self._update_blink(signals)
        else:
            self._update_turn(signals)
        return self.status

    def _update_blink(self, s: Signals) -> None:
        if self._stage == 0 and s.blink < EYES_OPEN:
            self._stage = 1
        elif self._stage == 1 and s.blink > EYES_CLOSED:
            self._stage = 2
        elif self._stage == 2 and s.blink < EYES_OPEN:
            self.status = Status.PASSED

    def _update_turn(self, s: Signals) -> None:
        wanted = 1.0 if self.challenge is Challenge.TURN_LEFT else -1.0
        if self._stage == 0:
            if abs(s.yaw) < FRONTAL_YAW:
                self._stage = 1
            return
        if s.yaw * wanted > TURN_YAW:
            self.status = Status.PASSED
        elif s.yaw * wanted < -TURN_YAW:
            self.status = Status.FAILED
            self.reason = "virou para o lado errado"

    def fail(self, reason: str) -> None:
        self.status = Status.FAILED
        self.reason = reason


def pick_challenges(count: int, rng: random.Random | None = None) -> list[Challenge]:
    """Sorteia desafios sem repetir; com 2 ou mais, no máximo um de virar a cabeça.

    Pedir "esquerda" e "direita" em sequência seria previsível para quem grava um vídeo.
    """
    rng = rng or random.SystemRandom()
    turn = rng.choice([Challenge.TURN_LEFT, Challenge.TURN_RIGHT])
    options = [Challenge.BLINK, turn]
    rng.shuffle(options)
    return options[: max(1, min(count, len(options)))]


# ---------------------------------------------------------------- anti-spoofing passivo

_FAS_MODELS = (
    ("minifasnet_v2_2.7_80x80.onnx", 2.7),
    ("minifasnet_v1se_4.0_80x80.onnx", 4.0),
)
_FAS_INPUT = 80


def crop_box(src_w: int, src_h: int, box: tuple[int, int, int, int], scale: float
             ) -> tuple[int, int, int, int]:
    """Janela ao redor do rosto ampliada por `scale`, deslocada para caber na imagem.

    Mesmo recorte usado no treino do MiniFASNet (CropImage._get_new_box no projeto original).
    """
    x, y, w, h = box
    w, h = max(w, 1), max(h, 1)
    scale = min((src_h - 1) / h, (src_w - 1) / w, scale)
    new_w, new_h = w * scale, h * scale
    cx, cy = x + w / 2, y + h / 2
    left, top = cx - new_w / 2, cy - new_h / 2
    right, bottom = cx + new_w / 2, cy + new_h / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1
    return int(left), int(top), int(right), int(bottom)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class PassiveAntiSpoof:
    """Probabilidade de o rosto ser real (0 a 1), média de dois modelos MiniFASNet."""

    def __init__(self) -> None:
        assets = resources.files("checkpoint") / "assets" / "minifasnet"
        self._nets = []
        for filename, scale in _FAS_MODELS:
            with resources.as_file(assets / filename) as path:
                self._nets.append((cv2.dnn.readNetFromONNX(str(path)), scale))

    def score(self, frame: np.ndarray, face: Face) -> float:
        h, w = frame.shape[:2]
        probs = []
        for net, scale in self._nets:
            left, top, right, bottom = crop_box(w, h, face.box, scale)
            patch = cv2.resize(frame[top:bottom + 1, left:right + 1], (_FAS_INPUT, _FAS_INPUT))
            blob = patch.astype(np.float32).transpose(2, 0, 1)[np.newaxis]  # BGR, 0-255
            net.setInput(blob)
            probs.append(float(_softmax(net.forward().reshape(-1))[1]))
        return float(np.mean(probs))


# ---------------------------------------------------------------- resultado da verificação


@dataclass
class LivenessResult:
    passed: bool
    reason: str = ""
    real_score: float = 0.0
    elapsed_s: float = 0.0
    challenges: list[Challenge] = field(default_factory=list)
