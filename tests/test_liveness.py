import random

import numpy as np
import pytest
from test_app import NOBODY, Clock, FakeEngine, face, frame, person

from checkpoint.app import verify_liveness
from checkpoint.config import Config
from checkpoint.gallery import Gallery, Match
from checkpoint.liveness import (
    Challenge,
    ChallengeTracker,
    PassiveAntiSpoof,
    Signals,
    Status,
    crop_box,
    pick_challenges,
    yaw_from_matrix,
)

OPEN, CLOSED = Signals(0.05, 0.0), Signals(0.9, 0.0)


def run(challenge: Challenge, signals: list[Signals]) -> ChallengeTracker:
    tracker = ChallengeTracker(challenge)
    for s in signals:
        tracker.update(s)
    return tracker


def test_blink_needs_open_closed_open():
    assert run(Challenge.BLINK, [OPEN, CLOSED, OPEN]).status is Status.PASSED
    assert run(Challenge.BLINK, [CLOSED, CLOSED]).status is Status.PENDING
    # Foto com olhos fechados não passa: nunca reabre.
    assert run(Challenge.BLINK, [OPEN, CLOSED]).status is Status.PENDING


def test_turn_needs_frontal_first_and_right_side():
    left = [Signals(0, 0), Signals(0, 12), Signals(0, 25)]
    assert run(Challenge.TURN_LEFT, left).status is Status.PASSED
    # Já começa virado: não conta.
    assert run(Challenge.TURN_LEFT, [Signals(0, 30)]).status is Status.PENDING
    wrong = run(Challenge.TURN_RIGHT, left)
    assert wrong.status is Status.FAILED and "errado" in wrong.reason
    assert run(Challenge.TURN_RIGHT, [Signals(0, 0), Signals(0, -25)]).status is Status.PASSED


def test_pick_challenges_never_asks_both_sides():
    rng = random.Random(1)
    for _ in range(50):
        two = pick_challenges(2, rng)
        assert len(two) == 2 and Challenge.BLINK in two
        assert len(pick_challenges(1, rng)) == 1
    assert {pick_challenges(1, random.Random(i))[0] for i in range(30)} == set(Challenge)


def test_yaw_sign_and_magnitude():
    angle = np.radians(30)
    turned = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                       [-np.sin(angle), 0, np.cos(angle)]])
    assert yaw_from_matrix(turned) == pytest.approx(30.0)
    assert yaw_from_matrix(np.eye(3)) == pytest.approx(0.0)


def test_crop_box_stays_inside_image():
    assert crop_box(640, 480, (270, 190, 100, 100), 2.7) == (185, 105, 455, 375)
    left, top, right, bottom = crop_box(640, 480, (0, 0, 150, 150), 4.0)
    assert left == 0 and top == 0 and right <= 639 and bottom <= 479
    left, top, right, bottom = crop_box(640, 480, (560, 400, 80, 80), 2.7)
    assert right == 639 and bottom == 479 and left >= 0 and top >= 0


def test_passive_model_outputs_probability():
    frame_ = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    score = PassiveAntiSpoof().score(frame_, face(150))
    assert 0.0 <= score <= 1.0


# ------------------------------------------------------------ fluxo completo com dublês


class FakeSensor:
    def __init__(self, signals):
        self.signals = list(signals)

    def measure(self, _frame, _ts):
        return self.signals.pop(0) if self.signals else OPEN


class FakeAntiSpoof:
    def __init__(self, score):
        self.value = score

    def score(self, _frame, _face):
        return self.value


def setup(rng_seed=0):
    gallery = Gallery()
    gallery.add(1, "Ana", [person(7)])
    gallery.add(2, "Bruno", [person(8)])
    employee = Match(1, "Ana", 0.9)
    # Com seed 0 e 1 desafio, o sorteio é sempre o mesmo: descobre qual para o teste.
    challenge = pick_challenges(1, random.Random(rng_seed))[0]
    return gallery, employee, challenge


def gesture(challenge):
    if challenge is Challenge.BLINK:
        return [OPEN, CLOSED, OPEN]
    side = 25 if challenge is Challenge.TURN_LEFT else -25
    return [Signals(0, 0), Signals(0, side), Signals(0, 0)]


def verify(signals, real=0.9, frames=None, seed=0):
    gallery, employee, _ = setup(seed)
    frames = frames or [frame(7)] * 30
    return verify_liveness(Config(), FakeEngine(), FakeSensor(signals), FakeAntiSpoof(real),
                           gallery, employee, iter(frames), preview=False, clock=Clock(),
                           rng=random.Random(seed))


def test_real_person_doing_gesture_passes():
    _, _, challenge = setup()
    result = verify(gesture(challenge) + [OPEN] * 5)
    assert result.passed, result.reason
    assert result.challenges == [challenge]
    assert result.real_score == pytest.approx(0.9)


def test_screen_or_photo_fails_passive_check():
    _, _, challenge = setup()
    result = verify(gesture(challenge) + [OPEN] * 5, real=0.2)
    assert not result.passed and "rosto real" in result.reason


def test_static_photo_times_out():
    result = verify([Signals(0.05, 30)] * 40)  # nunca de frente, nunca pisca
    assert not result.passed and result.reason == "tempo esgotado"


def test_other_person_during_check_fails():
    _, _, challenge = setup()
    frames = [frame(7), frame(8)] + [frame(7)] * 20
    result = verify(gesture(challenge), frames=frames)
    assert not result.passed and "Bruno" in result.reason


def test_face_leaving_fails():
    result = verify([OPEN] * 3, frames=[frame(7)] + [frame(NOBODY)] * 10)
    assert not result.passed and "saiu" in result.reason
