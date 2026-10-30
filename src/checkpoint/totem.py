"""Laço do totem: reconhece, faz a prova de vida e registra a marcação.

A alternativa sem biometria (tecla M) pede matrícula e PIN; a marcação fica com o método "pin",
para o RH saber que foi uma exceção.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta

import cv2
import numpy as np

from checkpoint import app
from checkpoint.app import GREEN, MANUAL, RED, SCREEN, YELLOW, draw, show, show_mirrored
from checkpoint.config import Config
from checkpoint.face import FaceEngine
from checkpoint.liveness import LandmarkSensor, LivenessResult, PassiveAntiSpoof
from checkpoint.punches import FACE, PIN, DuplicatePunch, Punch
from checkpoint.storage import Storage

log = logging.getLogger(__name__)

ENTRY_TIMEOUT_S = 20.0
ENTER_KEYS = (10, 13)
BACKSPACE_KEYS = (8, 127)
ESC = 27


@dataclass
class PinGuard:
    """Bloqueia a matrícula por alguns minutos depois de PINs errados seguidos."""

    max_attempts: int
    lockout_s: float
    clock: Callable[[], float] = time.monotonic
    _failures: dict[str, int] = field(default_factory=dict)
    _locked_until: dict[str, float] = field(default_factory=dict)

    def locked(self, code: str) -> bool:
        return self._locked_until.get(code, 0.0) > self.clock()

    def failure(self, code: str) -> None:
        self._failures[code] = self._failures.get(code, 0) + 1
        if self._failures[code] >= self.max_attempts:
            self._locked_until[code] = self.clock() + self.lockout_s
            self._failures[code] = 0

    def success(self, code: str) -> None:
        self._failures.pop(code, None)


def read_digits(
    frames: Iterator[np.ndarray], prompt: str, mask: bool = False, max_len: int = 12,
    clock: Callable[[], float] = time.monotonic,
) -> str | None:
    """Lê dígitos pelo teclado com a câmera ao fundo. Enter confirma; Esc cancela."""
    typed = ""
    start = clock()
    for frame in frames:
        if clock() - start > ENTRY_TIMEOUT_S:
            return None
        shown = "*" * len(typed) if mask else typed
        if not show_mirrored(frame, None, YELLOW, f"{prompt}: {shown}_",
                             "Enter confirma | Esc cancela"):
            return None
        key = SCREEN.last_key & 0xFF if SCREEN.last_key != -1 else -1
        if key == ESC:
            return None
        if key in ENTER_KEYS and typed:
            return typed
        if key in BACKSPACE_KEYS:
            typed = typed[:-1]
        elif ord("0") <= key <= ord("9") and len(typed) < max_len:
            typed += chr(key)
            start = clock()  # cada tecla renova o tempo
    return None


def show_qr(frames: Iterator[np.ndarray], url: str, text: str, seconds: float,
            clock: Callable[[], float] = time.monotonic) -> bool:
    """Mostra o QR do comprovante por alguns segundos. Qualquer tecla fecha; Q encerra (False)."""
    from checkpoint.receipt import qr_image

    start = clock()
    qr = None
    for frame in frames:
        if clock() - start >= seconds:
            return True
        view = cv2.flip(frame, 1)
        h, w = view.shape[:2]
        if qr is None:
            qr = qr_image(url, size=min(220, h - 120))
        side = qr.shape[0]
        x0, y0 = w - side - 16, (h - side) // 2
        view[y0:y0 + side, x0:x0 + side] = qr
        draw(view, None, GREEN, text)
        cv2.putText(view, "Comprovante: leia o QR com o celular", (x0 - 150, y0 + side + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if not show(view):
            return False
        if SCREEN.last_key != -1:
            return True
    return False


def describe(punch: Punch) -> str:
    verb = "registrada"
    return (f"{punch.kind_text} {verb} {punch.time_text} - {punch.employee_name} "
            f"(NSR {punch.nsr})")


def run_totem(
    cfg: Config,
    engine: FaceEngine,
    sensor: LandmarkSensor,
    antispoof: PassiveAntiSpoof,
    storage: Storage,
    frames: Iterator[np.ndarray],
    announce: Callable[[str], None] = print,
    receipt_url_for: Callable[[Punch], str] | None = None,
) -> None:
    """Atende funcionários até o operador sair (Q). Cada batida aprovada vira uma marcação.

    Com receipt_url_for, depois de cada registro o QR do comprovante aparece na tela.
    """
    gallery = storage.load_gallery()
    guard = PinGuard(cfg.punch.pin_max_attempts, cfg.punch.pin_lockout_min * 60)
    min_interval = timedelta(minutes=cfg.punch.min_interval_min)
    SCREEN.clock = True
    SCREEN.footer = "M = matricula e PIN"
    try:
        while True:
            found = app.wait_for_employee(cfg, engine, gallery, frames, allow_manual=True)
            if found is None:
                return

            if found == MANUAL:
                text, color, punch = _manual_punch(storage, guard, frames, min_interval,
                                                   announce)
            else:
                text, color, punch = _face_punch(cfg, engine, sensor, antispoof, storage,
                                                 gallery, found, frames, min_interval, announce)
            if text is None:
                return
            if punch is not None and receipt_url_for is not None:
                url = receipt_url_for(punch)
                announce(f"Comprovante: {url}")
                if not show_qr(frames, url, text, cfg.receipt.qr_seconds):
                    return
            if not app.wait_until_gone(engine, frames, text, color):
                return
    finally:
        SCREEN.clock = False
        SCREEN.footer = ""


def _face_punch(cfg, engine, sensor, antispoof, storage, gallery, employee, frames,
                min_interval, announce) -> tuple[str | None, tuple[int, int, int], Punch | None]:
    if cfg.liveness.enabled:
        result = app.verify_liveness(cfg, engine, sensor, antispoof, gallery, employee, frames)
    else:
        result = LivenessResult(True)
    if result.reason == "cancelado":
        return None, RED, None
    if not result.passed:
        announce(f"RECUSADO: {employee.name} | {result.reason} | "
                 f"rosto real {result.real_score:.2f}")
        return "Nao confirmado - tente de novo", RED, None

    code = next((e.code for e in storage.list_employees() if e.id == employee.employee_id),
                None)
    if code is None:  # removido enquanto o totem estava aberto
        return "Cadastro nao encontrado", RED, None
    return _register(storage, code, employee.name, FACE, min_interval, announce,
                     similarity=employee.score, real_score=result.real_score)


def _manual_punch(storage, guard, frames, min_interval, announce
                  ) -> tuple[str | None, tuple[int, int, int], Punch | None]:
    code = read_digits(frames, "Matricula")
    if code is None:
        return "Cancelado", YELLOW, None
    if guard.locked(code):
        return "Matricula bloqueada por alguns minutos", RED, None
    pin = read_digits(frames, "PIN", mask=True, max_len=8)
    if pin is None:
        return "Cancelado", YELLOW, None
    employee = storage.check_pin(code, pin)
    if employee is None:
        guard.failure(code)
        announce(f"PIN RECUSADO: matrícula {code}")
        return "Matricula ou PIN incorretos", RED, None
    guard.success(code)
    return _register(storage, employee.code, employee.name, PIN, min_interval, announce)


def _register(storage, code, name, method, min_interval, announce, similarity=None,
              real_score=None) -> tuple[str, tuple[int, int, int], Punch | None]:
    try:
        punch = storage.punches.register(code, name, method, similarity, real_score,
                                         min_interval=min_interval)
    except DuplicatePunch as exc:
        announce(f"DUPLICADO: {name} | {exc}")
        return f"{name}: ponto ja registrado as {exc.last.time_text}", YELLOW, None
    extra = " | via PIN" if method == PIN else f" | similaridade {similarity:.2f}"
    announce(f"REGISTRADO: {describe(punch)}{extra}")
    return describe(punch), GREEN, punch
