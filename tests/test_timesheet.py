from datetime import datetime, timedelta, timezone

import pytest

from checkpoint.punches import FACE, PIN
from checkpoint.storage import Storage
from checkpoint.timesheet import build, hours_text, punches_csv, timesheet_csv, totals

BRT = timezone(timedelta(hours=-3))


def at(hour: int, minute: int = 0, day: int = 30) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=BRT)


@pytest.fixture
def punches(tmp_path):
    with Storage.from_url(f"sqlite:///{(tmp_path / 'c.db').as_posix()}") as s:
        log = s.punches
        log.register("001", "Ana", FACE, 0.9, 0.98, now=at(8))
        log.register("002", "Bruno", PIN, now=at(9))
        log.register("001", "Ana", FACE, now=at(12))
        log.register("001", "Ana", FACE, now=at(13))
        log.register("001", "Ana", FACE, now=at(17, 30))
        log.register("001", "Ana", FACE, now=at(8, day=31))
        log.register("001", "Ana", FACE, now=at(12, 15, day=31))
        yield log.list()


def test_days_group_by_employee_and_date(punches):
    days = build(punches)
    assert [(d.employee_name, d.date) for d in days] == [
        ("Ana", "2026-10-30"), ("Ana", "2026-10-31"), ("Bruno", "2026-10-30")]
    ana, ana2, bruno = days
    assert ana.times == ["08:00", "12:00", "13:00", "17:30"]
    assert ana.complete and hours_text(ana.worked) == "08:30"
    assert hours_text(ana2.worked) == "04:15"
    assert not bruno.complete and bruno.worked == timedelta()


def test_totals_per_employee(punches):
    result = totals(build(punches))
    assert hours_text(result["001"]) == "12:45"
    assert result["002"] == timedelta()


def test_hours_text_beyond_a_day():
    assert hours_text(timedelta(hours=30, minutes=5, seconds=59)) == "30:05"


def test_csv_opens_in_excel(punches):
    sheet = timesheet_csv(build(punches))
    assert sheet.startswith("﻿Matrícula;Nome;Data")
    assert "001;Ana;2026-10-30;08:00 12:00 13:00 17:30;08:30;completo\r\n" in sheet
    assert "002;Bruno;2026-10-30;09:00;00:00;incompleto" in sheet

    raw = punches_csv(punches)
    lines = raw.strip().split("\r\n")
    assert len(lines) == 1 + len(punches)
    assert lines[1].startswith("1;001;Ana;2026-10-30;08:00:00;Entrada;face;0.90;0.98;")
    assert lines[2].split(";")[6:9] == ["pin", "", ""]
