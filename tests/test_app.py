import itertools

import numpy as np
import pytest

from checkpoint.app import EnrollError, benchmark, check_face, enroll, recognize
from checkpoint.config import Config
from checkpoint.face import Face, normalize
from checkpoint.gallery import Gallery

NOBODY = 0


def person(seed: int) -> np.ndarray:
    return normalize(np.random.default_rng(seed).standard_normal(128))


def face(size: int = 120) -> Face:
    raw = np.zeros(15, dtype=np.float32)
    raw[2:4] = size
    raw[14] = 0.95
    return Face(raw)


def frame(who: int, size: int = 120, faces: int = 1) -> np.ndarray:
    """Frame falso: o pixel (0,0) diz quem aparece, (0,1) o tamanho e (0,2) quantos rostos."""
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    f[0, 0, 0], f[0, 0, 1], f[0, 0, 2] = who, size, faces
    return f


class FakeEngine:
    def detect(self, f):
        who, size, count = (int(v) for v in f[0, 0])
        return [] if who == NOBODY else [face(size) for _ in range(count)]

    def embed(self, f, _face):
        return person(int(f[0, 0, 0]))


class Clock:
    """Relógio que avança 0,5 s a cada leitura, para não esperar o intervalo entre amostras."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.5
        return self.t


def test_check_face_reasons():
    assert check_face([], 80) == (None, "Nenhum rosto")
    assert check_face([face(), face()], 80)[1] == "Apenas uma pessoa por vez"
    assert check_face([face(40)], 80)[1] == "Aproxime-se da camera"
    assert check_face([face(120)], 80)[1] is None


def test_enroll_collects_configured_samples_and_skips_bad_frames():
    cfg = Config()
    frames = iter([frame(NOBODY), frame(7, size=40), frame(7, faces=2)]
                  + [frame(7)] * 20)
    samples = enroll(cfg, FakeEngine(), Gallery(), frames, preview=False, clock=Clock())
    assert len(samples) == cfg.recognition.enroll_samples
    assert all(np.allclose(s, person(7)) for s in samples)


def test_enroll_rejects_mixed_people():
    frames = iter([frame(7), frame(8)] + [frame(7)] * 10)
    with pytest.raises(EnrollError, match="mesma pessoa"):
        enroll(Config(), FakeEngine(), Gallery(), frames, preview=False, clock=Clock())


def test_enroll_rejects_face_already_registered():
    gallery = Gallery()
    gallery.add(1, "Ana", [person(7)])
    with pytest.raises(EnrollError, match="Ana"):
        enroll(Config(), FakeEngine(), gallery, iter([frame(7)] * 20), preview=False,
               clock=Clock())


def test_enroll_fails_when_camera_stops():
    with pytest.raises(EnrollError, match="parou"):
        enroll(Config(), FakeEngine(), Gallery(), iter([frame(7)] * 2), preview=False,
               clock=Clock())


def test_recognize_reports_only_changes():
    gallery = Gallery()
    gallery.add(1, "Ana", [person(7)])
    seen = []
    frames = [frame(7)] * 3 + [frame(NOBODY)] + [frame(9)] * 3 + [frame(7)] * 3
    recognize(Config(), FakeEngine(), gallery, iter(frames), preview=False,
              on_change=lambda m, best: seen.append((m and m.name, best.name)))
    # Desconhecido ainda informa o candidato mais parecido, para calibrar o limiar.
    assert seen == [("Ana", "Ana"), (None, "Ana"), ("Ana", "Ana")]


def test_benchmark_counts_frames():
    gallery = Gallery()
    gallery.add(1, "Ana", [person(7)])
    frames = itertools.cycle([frame(7), frame(NOBODY)])
    result = benchmark(FakeEngine(), gallery, frames, 10)
    assert (result.frames, result.frames_with_face) == (10, 5)
    assert "Total por frame" in result.report()


def test_recognize_ignores_isolated_frames():
    gallery = Gallery()
    gallery.add(1, "Ana", [person(7)])
    seen = []
    # Um frame "desconhecido" no meio de vários da Ana não muda a decisão.
    frames = [frame(7)] * 4 + [frame(9)] + [frame(7)] * 3
    recognize(Config(), FakeEngine(), gallery, iter(frames), preview=False,
              on_change=lambda m, best: seen.append(m and m.name))
    assert seen == ["Ana"]


def test_enroll_extra_samples_for_same_employee():
    gallery = Gallery()
    gallery.add(1, "Ana", [person(7)])
    samples = enroll(Config(), FakeEngine(), gallery, iter([frame(7)] * 20), preview=False,
                     clock=Clock(), same_employee_id=1)
    assert len(samples) == Config().recognition.enroll_samples
    with pytest.raises(EnrollError, match="Ana"):
        enroll(Config(), FakeEngine(), gallery, iter([frame(7)] * 20), preview=False,
               clock=Clock(), same_employee_id=2)
