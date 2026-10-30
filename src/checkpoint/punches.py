"""Marcações de ponto: registro só de inclusão, com NSR e hash SHA-256 encadeado.

Cada marcação guarda o hash da anterior. Alterar ou apagar qualquer linha quebra a cadeia, e
o comando `verificar` aponta onde. Além disso, gatilhos do SQLite recusam UPDATE e DELETE na
tabela, para que nem o próprio programa consiga editar uma marcação.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

GENESIS_HASH = "0" * 64

ENTRY = "entrada"
EXIT = "saida"

FACE = "face"
PIN = "pin"

SCHEMA = """
CREATE TABLE IF NOT EXISTS punches (
    nsr            INTEGER PRIMARY KEY,
    employee_code  TEXT NOT NULL,
    employee_name  TEXT NOT NULL,
    punched_at     TEXT NOT NULL,
    local_date     TEXT NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('entrada', 'saida')),
    method         TEXT NOT NULL CHECK (method IN ('face', 'pin')),
    similarity     REAL,
    real_score     REAL,
    prev_hash      TEXT NOT NULL,
    hash           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS punches_by_employee ON punches (employee_code, local_date);
CREATE TRIGGER IF NOT EXISTS punches_no_update BEFORE UPDATE ON punches
BEGIN
    SELECT RAISE(ABORT, 'marcações não podem ser alteradas');
END;
CREATE TRIGGER IF NOT EXISTS punches_no_delete BEFORE DELETE ON punches
BEGIN
    SELECT RAISE(ABORT, 'marcações não podem ser apagadas');
END;
"""

_COLUMNS = ("nsr, employee_code, employee_name, punched_at, local_date, kind, method, "
            "similarity, real_score, prev_hash, hash")


class PunchError(RuntimeError):
    pass


class DuplicatePunch(PunchError):
    def __init__(self, last: Punch):
        super().__init__(f"ponto já registrado às {last.time_text}")
        self.last = last


@dataclass(frozen=True)
class Punch:
    nsr: int
    employee_code: str
    employee_name: str
    punched_at: str  # ISO 8601 com fuso, ex.: 2026-10-30T08:01:23-03:00
    local_date: str  # AAAA-MM-DD no fuso do totem
    kind: str
    method: str
    similarity: float | None
    real_score: float | None
    prev_hash: str
    hash: str

    @property
    def moment(self) -> datetime:
        return datetime.fromisoformat(self.punched_at)

    @property
    def time_text(self) -> str:
        return self.moment.strftime("%H:%M:%S")

    @property
    def kind_text(self) -> str:
        return "Entrada" if self.kind == ENTRY else "Saída"


def compute_hash(nsr: int, employee_code: str, employee_name: str, punched_at: str,
                 kind: str, method: str, prev_hash: str) -> str:
    payload = "|".join([str(nsr), employee_code, employee_name, punched_at, kind, method,
                        prev_hash])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IntegrityReport:
    total: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


class PunchLog:
    def __init__(self, db: sqlite3.Connection):
        self._db = db
        self._db.executescript(SCHEMA)

    def _rows(self, where: str = "", params: tuple = ()) -> list[Punch]:
        rows = self._db.execute(
            f"SELECT {_COLUMNS} FROM punches {where} ORDER BY nsr", params
        ).fetchall()
        return [Punch(*row) for row in rows]

    def get(self, nsr: int) -> Punch | None:
        found = self._rows("WHERE nsr = ?", (nsr,))
        return found[0] if found else None

    def last_for(self, employee_code: str) -> Punch | None:
        row = self._db.execute(
            f"SELECT {_COLUMNS} FROM punches WHERE employee_code = ? ORDER BY nsr DESC LIMIT 1",
            (employee_code,),
        ).fetchone()
        return Punch(*row) if row else None

    def register(
        self,
        employee_code: str,
        employee_name: str,
        method: str,
        similarity: float | None = None,
        real_score: float | None = None,
        now: datetime | None = None,
        min_interval: timedelta = timedelta(minutes=2),
    ) -> Punch:
        """Grava a marcação. Entrada ou saída alterna a cada batida do mesmo dia.

        Recusa (DuplicatePunch) uma nova batida do mesmo funcionário antes de min_interval.
        """
        # Fuso do próprio totem, a menos que o horário já venha com fuso.
        now = now if now is not None and now.tzinfo else (now or datetime.now()).astimezone()
        punched_at = now.isoformat(timespec="seconds")
        local_date = now.date().isoformat()

        # BEGIN IMMEDIATE: NSR e hash anterior são lidos e gravados sem outra escrita no meio.
        self._db.execute("BEGIN IMMEDIATE")
        try:
            last = self.last_for(employee_code)
            if last is not None and now - last.moment < min_interval:
                raise DuplicatePunch(last)
            today = self._db.execute(
                "SELECT COUNT(*) FROM punches WHERE employee_code = ? AND local_date = ?",
                (employee_code, local_date),
            ).fetchone()[0]
            kind = ENTRY if today % 2 == 0 else EXIT
            tail = self._db.execute(
                "SELECT nsr, hash FROM punches ORDER BY nsr DESC LIMIT 1"
            ).fetchone()
            nsr, prev_hash = (tail[0] + 1, tail[1]) if tail else (1, GENESIS_HASH)
            digest = compute_hash(nsr, employee_code, employee_name, punched_at, kind, method,
                                  prev_hash)
            self._db.execute(
                f"INSERT INTO punches ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (nsr, employee_code, employee_name, punched_at, local_date, kind, method,
                 similarity, real_score, prev_hash, digest),
            )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return Punch(nsr, employee_code, employee_name, punched_at, local_date, kind, method,
                     similarity, real_score, prev_hash, digest)

    def list(self, day: date | None = None, employee_code: str | None = None,
             start: date | None = None, end: date | None = None) -> list[Punch]:
        """Marcações por dia exato ou intervalo [start, end], opcionalmente de um funcionário."""
        clauses, params = [], []
        if day is not None:
            clauses.append("local_date = ?")
            params.append(day.isoformat())
        if start is not None:
            clauses.append("local_date >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("local_date <= ?")
            params.append(end.isoformat())
        if employee_code is not None:
            clauses.append("employee_code = ?")
            params.append(employee_code)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._rows(where, tuple(params))

    def verify(self) -> IntegrityReport:
        """Confere NSR sequencial, encadeamento e hash de cada marcação."""
        problems = []
        punches = self._rows()
        prev_hash = GENESIS_HASH
        for expected_nsr, p in enumerate(punches, start=1):
            if p.nsr != expected_nsr:
                problems.append(f"NSR {expected_nsr} ausente (próximo encontrado: {p.nsr})")
            if p.prev_hash != prev_hash:
                problems.append(f"NSR {p.nsr}: não aponta para a marcação anterior")
            digest = compute_hash(p.nsr, p.employee_code, p.employee_name, p.punched_at,
                                  p.kind, p.method, p.prev_hash)
            if digest != p.hash:
                problems.append(f"NSR {p.nsr}: conteúdo alterado (hash não confere)")
            elif p.local_date != p.punched_at[:10]:
                problems.append(f"NSR {p.nsr}: data local não confere com o horário")
            prev_hash = p.hash
        return IntegrityReport(len(punches), problems)
