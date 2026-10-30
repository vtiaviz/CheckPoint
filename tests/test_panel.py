import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import numpy as np
import pytest

from checkpoint.face import normalize
from checkpoint.panel import (
    MAX_LOGIN_FAILURES,
    SESSION_COOKIE,
    SESSION_TTL_S,
    Panel,
    PanelError,
    Sessions,
    parse_period,
    set_password,
    start_panel,
)
from checkpoint.punches import FACE
from checkpoint.storage import Storage

BRT = timezone(timedelta(hours=-3))
PASSWORD = "senha-forte-123"
PERIOD = "de=2026-10-01&ate=2026-10-31"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


class Client:
    def __init__(self, base: str):
        self.base, self.cookie = base, ""

    def request(self, path: str, form: dict | None = None):
        data = urlencode(form).encode() if form is not None else None
        req = urllib.request.Request(self.base + path, data=data)
        if self.cookie:
            req.add_header("Cookie", self.cookie)
        try:
            resp = OPENER.open(req, timeout=5)
        except urllib.error.HTTPError as exc:
            resp = exc
        body = resp.read().decode("utf-8")
        set_cookie = resp.headers.get("Set-Cookie")
        if set_cookie:
            self.cookie = set_cookie.split(";", 1)[0]
        return resp.status, resp.headers, body

    def login(self, password: str = PASSWORD):
        return self.request("/entrar", {"senha": password})

    def csrf(self) -> str:
        _, _, body = self.request("/funcionarios")
        return re.search(r'name="csrf" value="([^"]+)"', body)[1]


@pytest.fixture
def server(tmp_path):
    url = f"sqlite:///{(tmp_path / 'c.db').as_posix()}"
    with Storage.from_url(url) as s:
        s.add_employee("001", "Ana <b>", [normalize(np.ones(128))])
        set_password(s, PASSWORD)
        s.punches.register("001", "Ana <b>", FACE, 0.9, 0.98,
                           now=datetime(2026, 10, 30, 8, 0, tzinfo=BRT))
        s.punches.register("001", "Ana <b>", FACE,
                           now=datetime(2026, 10, 30, 17, 0, tzinfo=BRT))
    panel = Panel(url)
    httpd = start_panel(panel, 0, "127.0.0.1")
    yield Client(f"http://127.0.0.1:{httpd.server_address[1]}"), url
    httpd.shutdown()
    httpd.server_close()


def test_password_minimum(tmp_path):
    url = f"sqlite:///{(tmp_path / 'c.db').as_posix()}"
    with Storage.from_url(url) as s, pytest.raises(PanelError):
        set_password(s, "curta")


def test_pages_require_login(server):
    client, _ = server
    for path in ("/", "/marcacoes", "/espelho", "/funcionarios", "/csv/marcacoes"):
        status, headers, _ = client.request(path)
        assert status == 303 and headers["Location"] == "/entrar"
    status, _, body = client.request("/entrar")
    assert status == 200 and 'name="senha"' in body


def test_wrong_password_then_login(server):
    client, _ = server
    _, _, body = client.login("errada")
    assert "Senha incorreta" in body and not client.cookie

    status, headers, _ = client.login()
    assert status == 303 and headers["Location"] == "/marcacoes"
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
    assert client.cookie.startswith(SESSION_COOKIE + "=")


def test_pages_render_and_escape(server):
    client, _ = server
    client.login()
    status, _, body = client.request(f"/marcacoes?{PERIOD}")
    assert status == 200 and "2 marcação(ões)" in body
    assert "Ana &lt;b&gt;" in body and "Ana <b>" not in body

    _, _, body = client.request(f"/espelho?{PERIOD}")
    assert "08:00 · 17:00" in body and "09:00" in body

    _, _, body = client.request("/integridade")
    assert "Íntegro: 2 marcações" in body

    status, _, _ = client.request("/nada")
    assert status == 404


def test_csv_export(server):
    client, _ = server
    client.login()
    status, headers, body = client.request(f"/csv/espelho?{PERIOD}")
    assert status == 200 and headers["Content-Type"].startswith("text/csv")
    assert 'filename="espelho-2026-10-01-a-2026-10-31.csv"' in headers["Content-Disposition"]
    assert "001;Ana <b>;2026-10-30;08:00 17:00;09:00;completo" in body

    _, _, body = client.request(f"/csv/marcacoes?{PERIOD}&matricula=999")
    assert body.strip() == "﻿" + body.strip()[1:] and body.count("\r\n") == 1


def test_post_needs_csrf(server):
    client, url = server
    client.login()
    form = {"matricula": "002", "nome": "Bruno", "pin": "1234"}
    status, _, _ = client.request("/funcionarios/novo", {**form, "csrf": "falso"})
    assert status == 403
    with Storage.from_url(url) as s:
        assert s.find_employee("002") is None


def test_employee_actions(server):
    client, url = server
    client.login()
    csrf = client.csrf()

    status, headers, _ = client.request(
        "/funcionarios/novo", {"matricula": "002", "nome": "Bruno", "pin": "1234", "csrf": csrf})
    assert status == 303 and headers["Location"].startswith("/funcionarios?ok=")
    with Storage.from_url(url) as s:
        assert s.check_pin("002", "1234").name == "Bruno"

    client.request("/funcionarios/pin", {"matricula": "001", "pin": "9876", "csrf": csrf})
    _, headers, _ = client.request("/funcionarios/pin",
                                   {"matricula": "001", "pin": "12", "csrf": csrf})
    assert headers["Location"].startswith("/funcionarios?erro=")
    _, _, body = client.request(headers["Location"])
    assert 'class="msg err"' in body

    client.request("/funcionarios/remover", {"matricula": "002", "csrf": csrf})
    with Storage.from_url(url) as s:
        assert s.check_pin("001", "9876") is not None
        assert s.find_employee("002") is None
        assert len(s.punches.list()) == 2


def test_logout_ends_session(server):
    client, _ = server
    client.login()
    csrf = client.csrf()
    old = client.cookie
    client.request("/sair", {"csrf": csrf})
    client.cookie = old  # mesmo reaproveitando o cookie antigo, a sessão acabou
    status, headers, _ = client.request("/marcacoes")
    assert status == 303 and headers["Location"] == "/entrar"


def test_login_lockout(server):
    client, _ = server
    for _ in range(MAX_LOGIN_FAILURES):
        client.login("errada")
    _, _, body = client.login()
    assert "Muitas tentativas" in body and not client.cookie


def test_sessions_expire():
    now = [0.0]
    sessions = Sessions(clock=lambda: now[0])
    token, csrf = sessions.create()
    assert sessions.csrf_for(token) == csrf
    now[0] = SESSION_TTL_S + 1
    assert sessions.csrf_for(token) is None
    assert sessions.csrf_for(None) is None


def test_parse_period():
    from datetime import date

    assert parse_period({"de": "2026-10-31", "ate": "2026-10-01"}) == (
        date(2026, 10, 1), date(2026, 10, 31))
    start, end = parse_period({"de": "lixo"})
    assert start.day == 1 and end == date.today()
