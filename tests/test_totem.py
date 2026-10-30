import numpy as np
import pytest

from checkpoint import app, totem
from checkpoint.app import MANUAL, SCREEN
from checkpoint.config import Config
from checkpoint.face import normalize
from checkpoint.gallery import Match
from checkpoint.liveness import LivenessResult
from checkpoint.storage import Storage
from checkpoint.totem import PinGuard, read_digits, run_totem


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_pin_guard_locks_after_failures():
    clock = Clock()
    guard = PinGuard(max_attempts=3, lockout_s=300, clock=clock)
    for _ in range(2):
        guard.failure("001")
    assert not guard.locked("001")
    guard.failure("001")
    assert guard.locked("001") and not guard.locked("002")
    clock.t = 301
    assert not guard.locked("001")


@pytest.fixture
def keys(monkeypatch):
    """Simula o teclado: cada frame "aperta" a próxima tecla da lista."""
    pressed = []

    def fake_show(frame, face, color, text, detail=""):
        fake_show.texts.append(text)
        SCREEN.last_key = pressed.pop(0) if pressed else -1
        return True

    fake_show.texts = []
    monkeypatch.setattr(totem, "show_mirrored", fake_show)
    return pressed, fake_show


def frames(n=50):
    return iter([np.zeros((4, 4, 3), np.uint8)] * n)


def test_read_digits_with_backspace_and_enter(keys):
    pressed, shown = keys
    pressed += [ord("1"), ord("2"), ord("x"), 8, ord("3"), 13]
    assert read_digits(frames(), "Matricula") == "13"
    assert shown.texts[-1] == "Matricula: 13_"


def test_read_digits_masks_and_cancels(keys):
    pressed, shown = keys
    pressed += [ord("9"), ord("9"), 27]
    assert read_digits(frames(), "PIN", mask=True) is None
    assert "PIN: **_" in shown.texts


def test_read_digits_ignores_enter_when_empty(keys):
    pressed, _ = keys
    pressed += [13, ord("5"), 13]
    assert read_digits(frames(), "Matricula") == "5"


# ------------------------------------------------------------ laço do totem com dublês


@pytest.fixture
def storage(tmp_path):
    with Storage(tmp_path / "checkpoint.db") as s:
        s.add_employee("001", "Ana", [normalize(np.ones(128))])
        s.set_pin("001", "1234")
        yield s


def run(storage, monkeypatch, visits, liveness=None, typed=()):
    """Roda o totem com uma fila de visitas (Match, MANUAL) e devolve o que foi anunciado."""
    queue = list(visits)
    monkeypatch.setattr(app, "wait_for_employee",
                        lambda *a, **k: queue.pop(0) if queue else None)
    monkeypatch.setattr(app, "verify_liveness",
                        lambda *a, **k: liveness or LivenessResult(True, real_score=0.97))
    monkeypatch.setattr(app, "wait_until_gone", lambda *a, **k: True)
    answers = list(typed)
    monkeypatch.setattr(totem, "read_digits", lambda *a, **k: answers.pop(0))
    said = []
    run_totem(Config(), None, None, None, storage, frames(), announce=said.append)
    assert SCREEN.clock is False  # restaura a tela ao sair
    return said


def ana():
    return Match(1, "Ana", 0.88)


def test_face_punch_registers_entry_then_blocks_duplicate(storage, monkeypatch):
    said = run(storage, monkeypatch, [ana(), ana()])
    assert said[0].startswith("REGISTRADO: Entrada registrada") and "NSR 1" in said[0]
    assert said[1].startswith("DUPLICADO: Ana")
    punch = storage.punches.list()[0]
    assert (punch.method, punch.similarity, punch.real_score) == ("face", 0.88, 0.97)


def test_failed_liveness_registers_nothing(storage, monkeypatch):
    said = run(storage, monkeypatch, [ana()],
               liveness=LivenessResult(False, "tempo esgotado", 0.0))
    assert said == ["RECUSADO: Ana | tempo esgotado | rosto real 0.00"]
    assert storage.punches.list() == []


def test_manual_punch_with_pin(storage, monkeypatch):
    said = run(storage, monkeypatch, [MANUAL], typed=["001", "1234"])
    assert "via PIN" in said[0]
    assert storage.punches.list()[0].method == "pin"


def test_wrong_pin_three_times_locks(storage, monkeypatch):
    typed = ["001", "0000"] * 3 + ["001"]
    said = run(storage, monkeypatch, [MANUAL] * 4, typed=typed)
    assert said == ["PIN RECUSADO: matrícula 001"] * 3
    assert storage.punches.list() == []
