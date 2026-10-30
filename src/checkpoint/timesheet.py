"""Espelho de ponto: marcações agrupadas por funcionário e dia, com horas trabalhadas."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from checkpoint.punches import ENTRY, Punch


@dataclass
class Day:
    employee_code: str
    employee_name: str
    date: str
    punches: list[Punch] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Toda entrada tem saída (número par de marcações)."""
        return len(self.punches) % 2 == 0

    @property
    def worked(self) -> timedelta:
        """Soma dos intervalos entrada → saída. Uma entrada sem saída não conta."""
        total = timedelta()
        ordered = sorted(self.punches, key=lambda p: p.nsr)
        for start, end in zip(ordered[::2], ordered[1::2]):
            if start.kind == ENTRY:
                total += end.moment - start.moment
        return total

    @property
    def times(self) -> list[str]:
        return [p.time_text[:5] for p in sorted(self.punches, key=lambda p: p.nsr)]


def build(punches: list[Punch]) -> list[Day]:
    """Um item por funcionário e dia, ordenado por nome e data."""
    days: dict[tuple[str, str], Day] = {}
    for p in punches:
        key = (p.employee_code, p.local_date)
        if key not in days:
            days[key] = Day(p.employee_code, p.employee_name, p.local_date)
        days[key].punches.append(p)
    return sorted(days.values(), key=lambda d: (d.employee_name.lower(), d.date))


def totals(days: list[Day]) -> dict[str, timedelta]:
    result: dict[str, timedelta] = defaultdict(timedelta)
    for day in days:
        result[day.employee_code] += day.worked
    return dict(result)


def hours_text(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _csv(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    # Ponto e vírgula e BOM: o Excel em português abre direto, com acentos corretos.
    writer = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    writer.writerows(rows)
    return "﻿" + buf.getvalue()


def timesheet_csv(days: list[Day]) -> str:
    rows = [["Matrícula", "Nome", "Data", "Marcações", "Horas trabalhadas", "Situação"]]
    for d in days:
        rows.append([d.employee_code, d.employee_name, d.date, " ".join(d.times),
                     hours_text(d.worked), "completo" if d.complete else "incompleto"])
    return _csv(rows)


def punches_csv(punches: list[Punch]) -> str:
    rows = [["NSR", "Matrícula", "Nome", "Data", "Hora", "Tipo", "Método", "Similaridade",
             "Rosto real", "Hash"]]
    for p in punches:
        rows.append([str(p.nsr), p.employee_code, p.employee_name, p.local_date, p.time_text,
                     p.kind_text, p.method, "" if p.similarity is None else f"{p.similarity:.2f}",
                     "" if p.real_score is None else f"{p.real_score:.2f}", p.hash])
    return _csv(rows)
