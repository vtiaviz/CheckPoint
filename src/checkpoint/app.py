"""Fluxos com câmera: cadastro, reconhecimento ao vivo e benchmark."""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from checkpoint.camera import open_camera
from checkpoint.config import Config
from checkpoint.face import Face, FaceEngine, normalize
from checkpoint.gallery import Gallery, Match
from checkpoint.models import DETECTOR, RECOGNIZER, ensure_model

log = logging.getLogger(__name__)

GREEN = (0, 200, 0)
RED = (0, 0, 220)
YELLOW = (0, 200, 220)
WINDOW = "CheckPoint (Q para sair)"

# Intervalo entre amostras no cadastro, para variar levemente pose e expressão.
SAMPLE_INTERVAL_S = 0.4
ENROLL_TIMEOUT_S = 45.0


class EnrollError(RuntimeError):
    pass


def build_engine(cfg: Config) -> FaceEngine:
    detector = ensure_model(DETECTOR, cfg.models.dir)
    recognizer = ensure_model(RECOGNIZER, cfg.models.dir)
    return FaceEngine(detector, recognizer, cfg.recognition.detect_score)


def camera_frames(cfg: Config) -> Iterator[np.ndarray]:
    cap = open_camera(cfg.camera)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                log.error("a câmera parou de entregar frames")
                return
            yield frame
    finally:
        cap.release()


def draw(frame: np.ndarray, face: Face | None, color: tuple[int, int, int], text: str) -> None:
    # As fontes do OpenCV não têm acentos: os textos da tela ficam sem acentuação.
    if face is not None:
        x, y, w, h = face.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def show(frame: np.ndarray) -> bool:
    """Mostra o frame; devolve False se o usuário pediu para sair."""
    cv2.imshow(WINDOW, frame)
    return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)


def check_face(faces: list[Face], min_size: int) -> tuple[Face | None, str | None]:
    """Rosto utilizável do frame, ou o motivo para não usar nenhum."""
    if not faces:
        return None, "Nenhum rosto"
    if len(faces) > 1:
        return faces[0], "Apenas uma pessoa por vez"
    if faces[0].size < min_size:
        return faces[0], "Aproxime-se da camera"
    return faces[0], None


def enroll(
    cfg: Config,
    engine: FaceEngine,
    gallery: Gallery,
    frames: Iterator[np.ndarray],
    preview: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> list[np.ndarray]:
    """Captura as amostras de um funcionário novo.

    Recusa se as amostras não parecerem da mesma pessoa ou se o rosto já estiver cadastrado.
    """
    try:
        samples = _collect_samples(cfg, engine, frames, preview, clock)
    finally:
        if preview:
            cv2.destroyAllWindows()

    threshold = cfg.recognition.match_threshold
    duplicate = gallery.identify(normalize(np.mean(samples, axis=0)), threshold)
    if duplicate is not None:
        raise EnrollError(
            f"este rosto já está cadastrado como {duplicate.name} "
            f"(similaridade {duplicate.score:.2f})"
        )
    return samples


def _collect_samples(
    cfg: Config,
    engine: FaceEngine,
    frames: Iterator[np.ndarray],
    preview: bool,
    clock: Callable[[], float],
) -> list[np.ndarray]:
    rec = cfg.recognition
    samples: list[np.ndarray] = []
    start = last = clock()
    last -= SAMPLE_INTERVAL_S

    for frame in frames:
        now = clock()
        if now - start > ENROLL_TIMEOUT_S:
            raise EnrollError("tempo esgotado sem capturar todas as amostras")

        face, problem = check_face(engine.detect(frame), rec.min_face_size)
        if problem is None and face is not None and now - last >= SAMPLE_INTERVAL_S:
            embedding = engine.embed(frame, face)
            if samples and float(normalize(np.mean(samples, axis=0)) @ embedding) < (
                rec.match_threshold
            ):
                raise EnrollError("as amostras não parecem ser da mesma pessoa; recomece")
            samples.append(embedding)
            last = now

        if preview:
            status = problem or f"Capturando {len(samples)}/{rec.enroll_samples}"
            draw(frame, face, YELLOW if problem else GREEN, status)
            if not show(frame):
                raise EnrollError("cadastro cancelado")

        if len(samples) >= rec.enroll_samples:
            break
    else:
        raise EnrollError("a câmera parou antes de terminar o cadastro")
    return samples


def recognize(
    cfg: Config,
    engine: FaceEngine,
    gallery: Gallery,
    frames: Iterator[np.ndarray],
    preview: bool = True,
    on_change: Callable[[Match | None, Match | None], None] | None = None,
) -> None:
    """Identifica quem está diante da câmera, quadro a quadro, até o usuário sair.

    on_change recebe o funcionário reconhecido (ou None) e o candidato mais parecido, mesmo
    abaixo do limiar, para ajudar a calibrar o match_threshold.
    """
    rec = cfg.recognition
    current: object = object()  # sentinela: força o primeiro aviso
    for frame in frames:
        face, problem = check_face(engine.detect(frame), rec.min_face_size)
        match = best = None
        if problem is None and face is not None:
            ranked = gallery.scores(engine.embed(frame, face))
            best = ranked[0] if ranked else None
            if best is not None and best.score >= rec.match_threshold:
                match = best

        if face is not None and problem is None:
            key = match.employee_id if match else None
            if key != current:
                current = key
                if on_change:
                    on_change(match, best)

        if preview:
            if problem:
                draw(frame, face, YELLOW, problem)
            elif match:
                draw(frame, face, GREEN, f"{match.name} ({match.score:.2f})")
            else:
                hint = f" (mais parecido: {best.score:.2f})" if best else ""
                draw(frame, face, RED, f"Nao reconhecido{hint}")
            if not show(frame):
                break
    if preview:
        cv2.destroyAllWindows()


@dataclass
class Benchmark:
    detect_ms: list[float] = field(default_factory=list)
    embed_ms: list[float] = field(default_factory=list)
    match_ms: list[float] = field(default_factory=list)
    frames: int = 0
    frames_with_face: int = 0

    @staticmethod
    def summary(values: list[float]) -> str:
        if not values:
            return "sem dados"
        p95 = sorted(values)[max(0, int(len(values) * 0.95) - 1)]
        return f"média {statistics.fmean(values):6.1f} ms | p95 {p95:6.1f} ms"

    def report(self) -> str:
        total = [
            d + e + m for d, e, m in zip(self.detect_ms, self.embed_ms, self.match_ms)
        ]
        lines = [
            f"Frames: {self.frames} ({self.frames_with_face} com rosto)",
            f"Detecção (YuNet):  {self.summary(self.detect_ms)}",
            f"Embedding (SFace): {self.summary(self.embed_ms)}",
            f"Busca 1:N:         {self.summary(self.match_ms)}",
            f"Total por frame:   {self.summary(total)}",
        ]
        if total:
            lines.append(f"Capacidade: ~{1000 / statistics.fmean(total):.0f} frames/s com rosto")
        return "\n".join(lines)


def benchmark(
    engine: FaceEngine, gallery: Gallery, frames: Iterator[np.ndarray], count: int
) -> Benchmark:
    result = Benchmark()
    for frame in frames:
        if result.frames >= count:
            break
        result.frames += 1
        t0 = time.perf_counter()
        faces = engine.detect(frame)
        t1 = time.perf_counter()
        if not faces:
            continue
        embedding = engine.embed(frame, faces[0])
        t2 = time.perf_counter()
        gallery.identify(embedding, 0.0)
        t3 = time.perf_counter()
        result.frames_with_face += 1
        result.detect_ms.append((t1 - t0) * 1000)
        result.embed_ms.append((t2 - t1) * 1000)
        result.match_ms.append((t3 - t2) * 1000)
    return result
