import numpy as np
import pytest

from checkpoint.face import normalize
from checkpoint.storage import Storage, StorageError, sqlite_path


def emb(seed: int) -> np.ndarray:
    return normalize(np.random.default_rng(seed).standard_normal(128))


@pytest.fixture
def storage(tmp_path):
    with Storage(tmp_path / "db" / "checkpoint.db") as s:
        yield s


def test_add_list_and_load_gallery(storage):
    storage.add_employee("001", "Ana", [emb(1), emb(2)])
    storage.add_employee("002", "Bruno", [emb(3)])
    assert [(e.code, e.name, e.samples) for e in storage.list_employees()] == [
        ("001", "Ana", 2), ("002", "Bruno", 1)]

    gallery = storage.load_gallery()
    assert len(gallery) == 2
    match = gallery.identify(emb(2), 0.99)
    assert match is not None and match.name == "Ana"
    assert match.score == pytest.approx(1.0, abs=1e-5)


def test_duplicate_code_rejected_without_partial_insert(storage):
    storage.add_employee("001", "Ana", [emb(1)])
    with pytest.raises(StorageError, match="já cadastrada"):
        storage.add_employee("001", "Outra", [emb(2)])
    assert [e.samples for e in storage.list_employees()] == [1]


def test_enroll_requires_samples(storage):
    with pytest.raises(StorageError, match="sem amostras"):
        storage.add_employee("001", "Ana", [])


def test_remove_deletes_face_vectors(storage):
    storage.add_employee("001", "Ana", [emb(1), emb(2)])
    assert storage.remove_employee("001") is True
    assert storage.remove_employee("001") is False
    assert storage.list_employees() == []
    assert storage._db.execute("SELECT COUNT(*) FROM face_samples").fetchone() == (0,)


def test_sqlite_url():
    assert str(sqlite_path("sqlite:///data/checkpoint.db")).replace("\\", "/") == (
        "data/checkpoint.db")
    with pytest.raises(StorageError, match="SQLite"):
        sqlite_path("postgresql://localhost/checkpoint")


def test_add_samples_to_existing_employee(storage):
    storage.add_employee("001", "Ana", [emb(1)])
    assert storage.add_samples("001", [emb(2), emb(3)]) == 3
    assert storage.find_employee("001").samples == 3
    with pytest.raises(StorageError, match="não encontrada"):
        storage.add_samples("999", [emb(4)])
