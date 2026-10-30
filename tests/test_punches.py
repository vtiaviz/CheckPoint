import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from checkpoint.face import normalize
from checkpoint.punches import ENTRY, EXIT, FACE, PIN, DuplicatePunch
from checkpoint.storage import Storage, StorageError

BRT = timezone(timedelta(hours=-3))


def at(hour: int, minute: int = 0, day: int = 30) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=BRT)


@pytest.fixture
def storage(tmp_path):
    with Storage(tmp_path / "checkpoint.db") as s:
        s.add_employee("001", "Ana", [normalize(np.ones(128))])
        yield s


def test_nsr_chain_and_entry_exit_alternation(storage):
    log = storage.punches
    p1 = log.register("001", "Ana", FACE, 0.9, 0.98, now=at(8))
    p2 = log.register("002", "Bruno", PIN, now=at(8, 5))
    p3 = log.register("001", "Ana", FACE, now=at(12))
    p4 = log.register("001", "Ana", FACE, now=at(8, day=31))  # dia seguinte: entrada
    assert [p.nsr for p in (p1, p2, p3, p4)] == [1, 2, 3, 4]
    assert [p.kind for p in (p1, p3, p4)] == [ENTRY, EXIT, ENTRY]
    assert p2.prev_hash == p1.hash and p3.prev_hash == p2.hash
    assert p1.punched_at == "2026-10-30T08:00:00-03:00" and p1.time_text == "08:00:00"
    assert log.verify().ok and log.verify().total == 4
    assert [p.nsr for p in log.list(day=at(8).date(), employee_code="001")] == [1, 3]


def test_duplicate_punch_within_interval(storage):
    storage.punches.register("001", "Ana", FACE, now=at(8))
    with pytest.raises(DuplicatePunch, match="08:00:00"):
        storage.punches.register("001", "Ana", FACE, now=at(8, 1))
    assert storage.punches.register("001", "Ana", FACE, now=at(8, 3)).nsr == 2


def test_database_refuses_update_and_delete(storage):
    storage.punches.register("001", "Ana", FACE, now=at(8))
    with pytest.raises(sqlite3.IntegrityError, match="alteradas"):
        storage._db.execute("UPDATE punches SET punched_at = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="apagadas"):
        storage._db.execute("DELETE FROM punches")


def test_verify_detects_tampering(storage):
    for hour in (8, 12, 13):
        storage.punches.register("001", "Ana", FACE, now=at(hour))
    # Alguém com acesso ao arquivo desliga os gatilhos e mexe nas linhas.
    db = storage._db
    db.execute("DROP TRIGGER punches_no_update")
    db.execute("DROP TRIGGER punches_no_delete")
    db.execute("UPDATE punches SET punched_at = '2026-10-30T07:30:00-03:00' WHERE nsr = 1")
    db.execute("DELETE FROM punches WHERE nsr = 2")
    db.commit()
    problems = storage.punches.verify().problems
    assert any("NSR 1: conteúdo alterado" in p for p in problems)
    assert any("NSR 2 ausente" in p for p in problems)
    assert any("NSR 3: não aponta" in p for p in problems)


def test_pin_set_check_and_validation(storage):
    assert storage.check_pin("001", "1234") is None  # sem PIN cadastrado
    storage.set_pin("001", "1234")
    assert storage.find_employee("001").has_pin
    assert storage.check_pin("001", "1234").name == "Ana"
    assert storage.check_pin("001", "0000") is None
    assert storage.check_pin("999", "1234") is None
    for bad in ("12", "abcd", "123456789"):
        with pytest.raises(StorageError, match="dígitos"):
            storage.set_pin("001", bad)
    with pytest.raises(StorageError, match="não encontrada"):
        storage.set_pin("999", "1234")
    stored = storage._db.execute("SELECT pin_hash FROM employees").fetchone()[0]
    assert "1234" not in stored and stored.startswith("scrypt$")


def test_removing_employee_keeps_punches(storage):
    storage.punches.register("001", "Ana", FACE, now=at(8))
    storage.remove_employee("001")
    assert [p.employee_name for p in storage.punches.list()] == ["Ana"]
    assert storage.punches.verify().ok


def test_migrates_database_from_m1(tmp_path):
    path = tmp_path / "antigo.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE employees (id INTEGER PRIMARY KEY, code TEXT NOT NULL UNIQUE,
                                name TEXT NOT NULL, created_at TEXT NOT NULL);
        INSERT INTO employees VALUES (1, '001', 'Ana', '2026-10-29T00:00:00+00:00');
    """)
    db.close()
    with Storage(path) as s:
        assert s.find_employee("001").has_pin is False
        s.set_pin("001", "4321")
        assert s.check_pin("001", "4321") is not None
