"""Fluxos com câmera: cadastro, reconhecimento ao vivo e benchmark."""

from __future__ import annotations

import logging
import random
import statistics
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from checkpoint.camera import open_camera
from checkpoint.config import Config
from checkpoint.face import Face, FaceEngine, normalize
from checkpoint.gallery import Gallery, Match
from checkpoint.liveness import (
    FRONTAL_YAW,
    ChallengeTracker,
    LandmarkSensor,
    LivenessResult,
    PassiveAntiSpoof,
    Status,
    pick_challenges,
)
from checkpoint.models import DETECTOR, RECOGNIZER, ensure_model

log = logging.getLogger(__name__)

GREEN = (0, 200, 0)
RED = (0, 0, 220)
YELLOW = (0, 200, 220)
WINDOW = "CheckPoint (Q para sair)"

# Intervalo entre amostras no cadastro, para variar levemente pose e expressão.
SAMPLE_INTERVAL_S = 0.4
ENROLL_TIMEOUT_S = 45.0

# Decisão por votação nos últimos frames: um frame borrado ou de perfil não troca o resultado.
VOTE_WINDOW = 5
VOTES_NEEDED = 3


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
    same_employee_id: int | None = None,
) -> list[np.ndarray]:
    """Captura as amostras de um funcionário novo.

    Recusa se as amostras não parecerem da mesma pessoa ou se o rosto já estiver cadastrado.
    Com same_employee_id (amostras extras de quem já existe), o rosto precisa ser dele.
    """
    try:
        samples = _collect_samples(cfg, engine, frames, preview, clock)
    finally:
        if preview:
            cv2.destroyAllWindows()

    threshold = cfg.recognition.match_threshold
    duplicate = gallery.identify(normalize(np.mean(samples, axis=0)), threshold)
    if same_employee_id is not None:
        if duplicate is not None and duplicate.employee_id != same_employee_id:
            raise EnrollError(
                f"este rosto parece ser de {duplicate.name} (similaridade {duplicate.score:.2f})"
            )
    elif duplicate is not None:
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
    """Identifica quem está diante da câmera até o usuário sair.

    O resultado só muda quando a mesma decisão aparece em VOTES_NEEDED dos últimos
    VOTE_WINDOW frames com rosto. on_change recebe o funcionário reconhecido (ou None) e o
    candidato mais parecido, mesmo abaixo do limiar, para ajudar a calibrar o match_threshold.
    """
    rec = cfg.recognition
    current: object = object()  # sentinela: força o primeiro aviso
    votes: deque[int | None] = deque(maxlen=VOTE_WINDOW)
    last_match: dict[int, Match] = {}
    for frame in frames:
        face, problem = check_face(engine.detect(frame), rec.min_face_size)
        match = best = None
        if problem is None and face is not None:
            ranked = gallery.scores(engine.embed(frame, face))
            best = ranked[0] if ranked else None
            if best is not None and best.score >= rec.match_threshold:
                match = best

        if face is None:
            votes.clear()  # pessoa saiu: a próxima começa do zero
        elif problem is None:
            key = match.employee_id if match else None
            if match:
                last_match[key] = match
            votes.append(key)
            winner, count = Counter(votes).most_common(1)[0]
            if count >= VOTES_NEEDED and winner != current:
                current = winner
                if on_change:
                    on_change(last_match[winner] if winner is not None else None, best)

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


# ---------------------------------------------------------------- batida com prova de vida

LEAVE_TIMEOUT_S = 1.0  # sem rosto por este tempo = a pessoa saiu do totem
MISSING_FACE_S = 1.5  # rosto sumiu durante a prova de vida


def show_mirrored(frame: np.ndarray, face: Face | None, color: tuple[int, int, int],
                  text: str, detail: str = "") -> bool:
    """Mostra a imagem espelhada, como num espelho: "esquerda" fica à esquerda da tela."""
    view = cv2.flip(frame, 1)
    if face is not None:
        x, y, w, h = face.box
        x = view.shape[1] - x - w
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
    draw(view, None, color, text)
    if detail:
        cv2.putText(view, detail, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1)
    return show(view)


def wait_for_employee(
    cfg: Config, engine: FaceEngine, gallery: Gallery, frames: Iterator[np.ndarray],
    preview: bool = True,
) -> Match | None:
    """Espera até alguém ser reconhecido de forma estável (votação entre frames).

    Devolve None se a câmera parar ou o usuário sair.
    """
    rec = cfg.recognition
    votes: deque[int | None] = deque(maxlen=VOTE_WINDOW)
    last: dict[int, Match] = {}
    for frame in frames:
        face, problem = check_face(engine.detect(frame), rec.min_face_size)
        if face is None:
            votes.clear()
        elif problem is None:
            match = gallery.identify(engine.embed(frame, face), rec.match_threshold)
            key = match.employee_id if match else None
            if match:
                last[key] = match
            votes.append(key)
            winner, count = Counter(votes).most_common(1)[0]
            if winner is not None and count >= VOTES_NEEDED:
                return last[winner]
        if preview:
            text = problem or ("Identificando..." if face is not None else "Aproxime-se")
            if not show_mirrored(frame, face, YELLOW, text):
                return None
    return None


def verify_liveness(
    cfg: Config,
    engine: FaceEngine,
    sensor: LandmarkSensor,
    antispoof: PassiveAntiSpoof,
    gallery: Gallery,
    employee: Match,
    frames: Iterator[np.ndarray],
    preview: bool = True,
    clock: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
) -> LivenessResult:
    """Prova de vida de quem acabou de ser reconhecido.

    Aprova quando todos os desafios sorteados forem cumpridos, a mediana do anti-spoofing
    passivo nos frames de frente passar do limiar e ninguém mais for reconhecido no meio.
    """
    live, rec = cfg.liveness, cfg.recognition
    challenges = pick_challenges(live.challenges, rng)
    trackers = [ChallengeTracker(c) for c in challenges]
    real_scores: list[float] = []
    start = last_face = clock()

    def result(passed: bool, reason: str = "") -> LivenessResult:
        score = float(np.median(real_scores)) if real_scores else 0.0
        return LivenessResult(passed, reason, score, clock() - start, challenges)

    for frame in frames:
        now = clock()
        if now - start > live.timeout_s:
            return result(False, "tempo esgotado")
        current = next((t for t in trackers if t.status is Status.PENDING), None)

        face, problem = check_face(engine.detect(frame), rec.min_face_size)
        if face is None:
            if now - last_face > MISSING_FACE_S:
                return result(False, "o rosto saiu da câmera")
        else:
            last_face = now
        if problem == "Apenas uma pessoa por vez":
            return result(False, "mais de uma pessoa diante da câmera")

        signals = None
        if face is not None and problem is None:
            ranked = gallery.scores(engine.embed(frame, face))
            other = next((m for m in ranked if m.employee_id != employee.employee_id), None)
            if other is not None and other.score >= rec.match_threshold:
                return result(False, f"outra pessoa apareceu ({other.name})")

            signals = sensor.measure(frame, int(now * 1000))
            if signals is not None and abs(signals.yaw) < FRONTAL_YAW:
                # O anti-spoofing foi treinado com rostos de frente.
                real_scores.append(antispoof.score(frame, face))
            if signals is not None and current is not None:
                if current.update(signals) is Status.FAILED:
                    return result(False, current.reason)
                current = next((t for t in trackers if t.status is Status.PENDING), None)

        if current is None and len(real_scores) >= live.min_passive_frames:
            score = float(np.median(real_scores))
            if score < live.real_threshold:
                return result(False, f"imagem não parece de um rosto real ({score:.2f})")
            return result(True)

        if preview:
            text = current.challenge.screen_text if current else "Olhe para a camera"
            detail = f"{employee.name} - {now - start:.1f}s"
            if not show_mirrored(frame, face, YELLOW if problem else GREEN, text, detail):
                return result(False, "cancelado")
    return result(False, "a câmera parou")


def wait_until_gone(engine: FaceEngine, frames: Iterator[np.ndarray], text: str,
                    color: tuple[int, int, int], preview: bool = True,
                    clock: Callable[[], float] = time.monotonic) -> bool:
    """Mantém o resultado na tela até a pessoa sair; False se o usuário pediu para sair."""
    gone_since: float | None = None
    for frame in frames:
        faces = engine.detect(frame)
        now = clock()
        if faces:
            gone_since = None
        elif gone_since is None:
            gone_since = now
        elif now - gone_since >= LEAVE_TIMEOUT_S:
            return True
        if preview and not show_mirrored(frame, faces[0] if faces else None, color, text):
            return False
    return False
