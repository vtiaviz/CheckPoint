"""Cadastro de funcionários, embeddings, PINs e marcações em SQLite.

Os vetores ainda são gravados em claro; a criptografia (AES-256-GCM) entra no M6.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from checkpoint.face import EMBEDDING_SIZE
from checkpoint.gallery import Gallery
from checkpoint.punches import PunchLog

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id          INTEGER PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    pin_hash    TEXT
);
CREATE TABLE IF NOT EXISTS face_samples (
    id           INTEGER PRIMARY KEY,
    employee_id  INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    embedding    BLOB NOT NULL,
    created_at   TEXT NOT NULL
);
"""


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Employee:
    id: int
    code: str
    name: str
    samples: int
    has_pin: bool = False


def sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise StorageError(f"só SQLite é suportado no protótipo: {database_url}")
    return Path(database_url[len(prefix):])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(SCHEMA)
        self._migrate()
        self.punches = PunchLog(self._db)

    def _migrate(self) -> None:
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(employees)")}
        if "pin_hash" not in columns:  # bancos criados no M1/M2
            with self._db:
                self._db.execute("ALTER TABLE employees ADD COLUMN pin_hash TEXT")

    @classmethod
    def from_url(cls, database_url: str) -> Storage:
        return cls(sqlite_path(database_url))

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Storage:  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add_employee(self, code: str, name: str, embeddings: list[np.ndarray]) -> int:
        """Cadastra o funcionário com suas amostras numa única transação."""
        if not embeddings:
            raise StorageError("cadastro sem amostras de rosto")
        try:
            with self._db:
                cur = self._db.execute(
                    "INSERT INTO employees (code, name, created_at) VALUES (?, ?, ?)",
                    (code, name, _now()),
                )
                employee_id = int(cur.lastrowid)
                self._db.executemany(
                    "INSERT INTO face_samples (employee_id, embedding, created_at) "
                    "VALUES (?, ?, ?)",
                    [(employee_id, _to_blob(e), _now()) for e in embeddings],
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"matrícula {code} já cadastrada") from exc
        return employee_id

    def find_employee(self, code: str) -> Employee | None:
        return next((e for e in self.list_employees() if e.code == code), None)

    def add_samples(self, code: str, embeddings: list[np.ndarray]) -> int:
        """Acrescenta amostras a um funcionário existente; devolve o total de amostras."""
        employee = self.find_employee(code)
        if employee is None:
            raise StorageError(f"matrícula {code} não encontrada")
        if not embeddings:
            raise StorageError("nenhuma amostra para acrescentar")
        with self._db:
            self._db.executemany(
                "INSERT INTO face_samples (employee_id, embedding, created_at) VALUES (?, ?, ?)",
                [(employee.id, _to_blob(e), _now()) for e in embeddings],
            )
        return employee.samples + len(embeddings)

    def set_pin(self, code: str, pin: str) -> None:
        validate_pin(pin)
        with self._db:
            cur = self._db.execute("UPDATE employees SET pin_hash = ? WHERE code = ?",
                                   (hash_pin(pin), code))
        if cur.rowcount == 0:
            raise StorageError(f"matrícula {code} não encontrada")

    def check_pin(self, code: str, pin: str) -> Employee | None:
        """Funcionário dono da matrícula se o PIN conferir; senão None."""
        row = self._db.execute("SELECT pin_hash FROM employees WHERE code = ?",
                               (code,)).fetchone()
        if row is None or row[0] is None or not verify_pin(pin, row[0]):
            return None
        return self.find_employee(code)

    def remove_employee(self, code: str) -> bool:
        """Remove o funcionário, seus vetores faciais e o PIN (LGPD: desligamento).

        As marcações de ponto ficam: são registro trabalhista, sem dado biométrico.
        """
        with self._db:
            cur = self._db.execute("DELETE FROM employees WHERE code = ?", (code,))
        return cur.rowcount > 0

    def list_employees(self) -> list[Employee]:
        rows = self._db.execute(
            "SELECT e.id, e.code, e.name, COUNT(s.id), e.pin_hash IS NOT NULL FROM employees e "
            "LEFT JOIN face_samples s ON s.employee_id = e.id "
            "GROUP BY e.id ORDER BY e.name"
        ).fetchall()
        return [Employee(i, c, n, s, bool(p)) for i, c, n, s, p in rows]

    def load_gallery(self) -> Gallery:
        gallery = Gallery()
        rows = self._db.execute(
            "SELECT e.id, e.name, s.embedding FROM employees e "
            "JOIN face_samples s ON s.employee_id = e.id ORDER BY e.id"
        ).fetchall()
        grouped: dict[int, tuple[str, list[np.ndarray]]] = {}
        for employee_id, name, blob in rows:
            grouped.setdefault(employee_id, (name, []))[1].append(_from_blob(blob))
        for employee_id, (name, embeddings) in grouped.items():
            gallery.add(employee_id, name, embeddings)
        return gallery


def _to_blob(embedding: np.ndarray) -> bytes:
    vector = np.asarray(embedding, dtype="<f4").reshape(-1)
    if vector.size != EMBEDDING_SIZE:
        raise StorageError(f"embedding com {vector.size} posições, esperado {EMBEDDING_SIZE}")
    return vector.tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").astype(np.float32)


# ---------------------------------------------------------------- PIN

PIN_MIN, PIN_MAX = 4, 8
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def validate_pin(pin: str) -> None:
    if not (pin.isdigit() and PIN_MIN <= len(pin) <= PIN_MAX):
        raise StorageError(f"o PIN deve ter de {PIN_MIN} a {PIN_MAX} dígitos")


def hash_pin(pin: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_pin(pin: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(pin.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)
