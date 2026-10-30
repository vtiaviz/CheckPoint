"""Painel web do RH: funcionários, marcações, espelho de ponto e exportação CSV.

Roda separado do totem (comando `painel`), lendo o mesmo banco. O acesso exige a senha do
painel; a sessão fica num cookie HttpOnly/SameSite=Strict e todo formulário leva um token
contra CSRF. Errar a senha várias vezes bloqueia o IP por alguns minutos.

O cadastro do rosto continua no totem (precisa da câmera); pelo painel dá para cadastrar
funcionários só com PIN, definir ou apagar PIN e remover funcionários.
"""

from __future__ import annotations

import html
import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlencode, urlparse

from checkpoint.storage import Storage, StorageError, hash_pin, verify_pin
from checkpoint.timesheet import build, hours_text, punches_csv, timesheet_csv, totals

log = logging.getLogger(__name__)

PASSWORD_KEY = "panel_password"
SESSION_COOKIE = "checkpoint_rh"
SESSION_TTL_S = 8 * 3600
MAX_LOGIN_FAILURES = 5
LOGIN_LOCK_S = 300
MIN_PASSWORD = 8


class PanelError(RuntimeError):
    pass


def set_password(storage: Storage, password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise PanelError(f"a senha do painel deve ter pelo menos {MIN_PASSWORD} caracteres")
    storage.set_setting(PASSWORD_KEY, hash_pin(password))


@dataclass
class Sessions:
    clock: Callable[[], float] = time.monotonic
    _items: dict[str, tuple[float, str]] = field(default_factory=dict)
    _failures: dict[str, tuple[int, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self) -> tuple[str, str]:
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(16)
        with self._lock:
            self._items[token] = (self.clock() + SESSION_TTL_S, csrf)
        return token, csrf

    def csrf_for(self, token: str | None) -> str | None:
        with self._lock:
            item = self._items.get(token or "")
            if item is None or item[0] < self.clock():
                self._items.pop(token or "", None)
                return None
            return item[1]

    def drop(self, token: str | None) -> None:
        with self._lock:
            self._items.pop(token or "", None)

    def locked(self, ip: str) -> bool:
        count, until = self._failures.get(ip, (0, 0.0))
        return count >= MAX_LOGIN_FAILURES and until > self.clock()

    def failed(self, ip: str) -> None:
        count, until = self._failures.get(ip, (0, 0.0))
        if until and until < self.clock():
            count = 0
        count += 1
        self._failures[ip] = (count, self.clock() + LOGIN_LOCK_S)

    def succeeded(self, ip: str) -> None:
        self._failures.pop(ip, None)


# ---------------------------------------------------------------- HTML

CSS = """
:root { --bg:#f6f7f8; --fg:#1d2228; --muted:#5d6670; --line:#d9dde1; --accent:#1f6f43;
        --warn:#9a5b00; --bad:#a32020; --card:#fff; }
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--fg); }
header { background:var(--fg); color:#fff; padding:10px 16px; display:flex; gap:16px;
         align-items:center; flex-wrap:wrap; }
header a { color:#fff; text-decoration:none; opacity:.85; } header a.on { opacity:1;
         font-weight:600; border-bottom:2px solid #fff; }
header form { margin-left:auto; }
main { padding:16px; max-width:1100px; margin:0 auto; }
h1 { font-size:1.25rem; } .card { background:var(--card); border:1px solid var(--line);
     border-radius:8px; padding:12px; margin-bottom:16px; overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.92rem; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
         white-space:nowrap; }
th { color:var(--muted); font-weight:600; } td.num { font-variant-numeric: tabular-nums; }
form.inline { display:inline; } input, select, button { font:inherit; padding:6px 8px;
     border:1px solid var(--line); border-radius:6px; }
button { background:var(--accent); color:#fff; border-color:var(--accent); cursor:pointer; }
button.danger { background:var(--bad); border-color:var(--bad); }
button.plain { background:#fff; color:var(--fg); }
.ok { color:var(--accent); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.warn { color:var(--warn); } .muted { color:var(--muted); font-size:.85rem; }
.filters { display:flex; gap:8px; flex-wrap:wrap; align-items:end; }
.filters label { display:flex; flex-direction:column; font-size:.8rem; color:var(--muted); }
.msg { padding:8px 12px; border-radius:6px; background:#e8f3ec; margin-bottom:12px; }
.msg.err { background:#f8e6e6; }
"""

NAV = [("/marcacoes", "Marcações"), ("/espelho", "Espelho de ponto"),
       ("/funcionarios", "Funcionários"), ("/integridade", "Integridade")]


def esc(value: object) -> str:
    return html.escape(str(value))


def page(title: str, body: str, path: str = "", csrf: str | None = None) -> str:
    nav = ""
    if csrf is not None:
        links = "".join(
            f'<a href="{href}" class="{"on" if path == href else ""}">{label}</a>'
            for href, label in NAV
        )
        nav = (f"<strong>CheckPoint RH</strong>{links}"
               f'<form method="post" action="/sair">{csrf_field(csrf)}'
               '<button class="plain">Sair</button></form>')
    return (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)} - CheckPoint RH</title><style>{CSS}</style></head>"
            f"<body><header>{nav or '<strong>CheckPoint RH</strong>'}</header>"
            f"<main>{body}</main></body></html>")


def csrf_field(csrf: str) -> str:
    return f'<input type="hidden" name="csrf" value="{esc(csrf)}">'


def message(params: dict[str, str]) -> str:
    if params.get("ok"):
        return f'<div class="msg">{esc(params["ok"])}</div>'
    if params.get("erro"):
        return f'<div class="msg err">{esc(params["erro"])}</div>'
    return ""


def parse_period(params: dict[str, str]) -> tuple[date, date]:
    today = date.today()
    try:
        start = date.fromisoformat(params["de"]) if params.get("de") else today.replace(day=1)
        end = date.fromisoformat(params["ate"]) if params.get("ate") else today
    except ValueError:
        start, end = today.replace(day=1), today
    if end < start:
        start, end = end, start
    return start, min(end, start + timedelta(days=366))


def filter_form(action: str, start: date, end: date, code: str, employees, csv_href: str) -> str:
    options = '<option value="">Todos</option>' + "".join(
        f'<option value="{esc(e.code)}" {"selected" if e.code == code else ""}>'
        f"{esc(e.name)} ({esc(e.code)})</option>" for e in employees
    )
    return (f'<form class="filters card" method="get" action="{action}">'
            f'<label>De<input type="date" name="de" value="{start}"></label>'
            f'<label>Até<input type="date" name="ate" value="{end}"></label>'
            f'<label>Funcionário<select name="matricula">{options}</select></label>'
            f'<button>Filtrar</button><a href="{csv_href}">Exportar CSV</a></form>')


# ---------------------------------------------------------------- aplicação


@dataclass
class Panel:
    database_url: str
    sessions: Sessions = field(default_factory=Sessions)

    def storage(self) -> Storage:
        # Uma conexão por requisição: o sqlite3 não compartilha conexões entre threads.
        return Storage.from_url(self.database_url)

    def password_hash(self) -> str | None:
        with self.storage() as s:
            return s.get_setting(PASSWORD_KEY)

    # Cada handler devolve (status, content_type, corpo, cabeçalhos extras).

    def login_page(self, error: str = "") -> tuple[int, str, str, dict]:
        configured = self.password_hash() is not None
        warn = "" if configured else (
            '<div class="msg err">Senha do painel ainda não definida. No computador do '
            "totem, rode: <code>python -m checkpoint senha-painel</code></div>")
        err = f'<div class="msg err">{esc(error)}</div>' if error else ""
        body = (f'<h1>Entrar</h1>{warn}{err}<form class="card" method="post" action="/entrar">'
                '<label>Senha do painel<br><input type="password" name="senha" autofocus '
                'required></label> <button>Entrar</button></form>')
        return 200, "text/html", page("Entrar", body), {}

    def login(self, ip: str, form: dict[str, str]) -> tuple[int, str, str, dict]:
        stored = self.password_hash()
        if self.sessions.locked(ip):
            return self.login_page("Muitas tentativas. Aguarde alguns minutos.")
        if stored is None or not verify_pin(form.get("senha", ""), stored):
            self.sessions.failed(ip)
            log.warning("login do painel recusado de %s", ip)
            return self.login_page("Senha incorreta.")
        self.sessions.succeeded(ip)
        token, _ = self.sessions.create()
        cookie = (f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; "
                  f"Max-Age={SESSION_TTL_S}")
        return 303, "text/plain", "", {"Location": "/marcacoes", "Set-Cookie": cookie}

    def punches_page(self, params, csrf) -> str:
        start, end = parse_period(params)
        code = params.get("matricula", "")
        with self.storage() as s:
            employees = s.list_employees()
            punches = s.punches.list(start=start, end=end, employee_code=code or None)
        rows = "".join(
            f"<tr><td class=num>{p.nsr}</td><td>{esc(p.local_date)}</td>"
            f"<td class=num>{p.time_text}</td><td>{p.kind_text}</td>"
            f"<td>{esc(p.employee_name)}</td><td>{esc(p.employee_code)}</td>"
            f"<td>{'PIN' if p.method == 'pin' else 'rosto'}</td></tr>"
            for p in reversed(punches)
        ) or '<tr><td colspan="7" class="muted">Nenhuma marcação no período.</td></tr>'
        query = urlencode({"de": start, "ate": end, "matricula": code})
        body = (f"<h1>Marcações</h1>{message(params)}"
                f"{filter_form('/marcacoes', start, end, code, employees, '/csv/marcacoes?' + query)}"
                f'<div class="card"><p class="muted">{len(punches)} marcação(ões), mais recentes '
                "primeiro.</p><table><tr><th>NSR</th><th>Data</th><th>Hora</th><th>Tipo</th>"
                f"<th>Nome</th><th>Matrícula</th><th>Via</th></tr>{rows}</table></div>")
        return page("Marcações", body, "/marcacoes", csrf)

    def timesheet_page(self, params, csrf) -> str:
        start, end = parse_period(params)
        code = params.get("matricula", "")
        with self.storage() as s:
            employees = s.list_employees()
            days = build(s.punches.list(start=start, end=end, employee_code=code or None))
        names = {d.employee_code: d.employee_name for d in days}
        rows = "".join(
            f"<tr><td>{esc(d.employee_name)}</td><td>{esc(d.date)}</td>"
            f"<td class=num>{' · '.join(d.times)}</td><td class=num>{hours_text(d.worked)}</td>"
            f"<td>{'' if d.complete else '<span class=warn>sem saída</span>'}</td></tr>"
            for d in days
        ) or '<tr><td colspan="5" class="muted">Nenhuma marcação no período.</td></tr>'
        total_rows = "".join(
            f"<tr><td>{esc(names[c])}</td><td>{esc(c)}</td><td class=num>{hours_text(t)}</td></tr>"
            for c, t in sorted(totals(days).items(), key=lambda i: names[i[0]].lower())
        )
        query = urlencode({"de": start, "ate": end, "matricula": code})
        body = (f"<h1>Espelho de ponto</h1>"
                f"{filter_form('/espelho', start, end, code, employees, '/csv/espelho?' + query)}"
                f'<div class="card"><h2 class="muted">Total no período</h2><table><tr>'
                f"<th>Nome</th><th>Matrícula</th><th>Horas</th></tr>{total_rows}</table></div>"
                f'<div class="card"><table><tr><th>Nome</th><th>Data</th><th>Marcações</th>'
                f"<th>Horas</th><th></th></tr>{rows}</table>"
                '<p class="muted">Dias "sem saída" somam só os pares completos de entrada e '
                "saída.</p></div>")
        return page("Espelho de ponto", body, "/espelho", csrf)

    def employees_page(self, params, csrf) -> str:
        with self.storage() as s:
            employees = s.list_employees()
        rows = "".join(
            f"<tr><td>{esc(e.code)}</td><td>{esc(e.name)}</td>"
            f"<td>{'rosto (' + str(e.samples) + ' amostras)' if e.samples else '<span class=warn>sem rosto</span>'}</td>"
            f"<td>{'sim' if e.has_pin else 'não'}</td><td>"
            f'<form class="inline" method="post" action="/funcionarios/pin">{csrf_field(csrf)}'
            f'<input type="hidden" name="matricula" value="{esc(e.code)}">'
            '<input name="pin" inputmode="numeric" pattern="[0-9]{4,8}" placeholder="novo PIN" '
            'size="9" required> <button class="plain">Definir PIN</button></form> '
            f'<form class="inline" method="post" action="/funcionarios/remover" '
            f"onsubmit=\"return confirm('Remover {esc(e.name)}? O rosto e o PIN serão apagados; "
            f"as marcações ficam.')\">{csrf_field(csrf)}"
            f'<input type="hidden" name="matricula" value="{esc(e.code)}">'
            '<button class="danger">Remover</button></form></td></tr>'
            for e in employees
        ) or '<tr><td colspan="5" class="muted">Nenhum funcionário cadastrado.</td></tr>'
        body = (f"<h1>Funcionários</h1>{message(params)}"
                f'<div class="card"><table><tr><th>Matrícula</th><th>Nome</th><th>Biometria</th>'
                f"<th>PIN</th><th>Ações</th></tr>{rows}</table></div>"
                f'<form class="card filters" method="post" action="/funcionarios/novo">'
                f"{csrf_field(csrf)}<strong>Cadastrar só com PIN (sem biometria)</strong>"
                '<label>Matrícula<input name="matricula" required></label>'
                '<label>Nome<input name="nome" required></label>'
                '<label>PIN<input name="pin" inputmode="numeric" pattern="[0-9]{4,8}" '
                'required></label><button>Cadastrar</button></form>'
                '<p class="muted">O rosto é cadastrado no totem: <code>python -m checkpoint '
                'cadastrar MATRICULA "Nome"</code> (ou <code>--adicionar</code> para quem já '
                "existe aqui).</p>")
        return page("Funcionários", body, "/funcionarios", csrf)

    def integrity_page(self, params, csrf) -> str:
        with self.storage() as s:
            report = s.punches.verify()
        if report.ok:
            status = (f'<p class="ok">Íntegro: {report.total} marcações, NSR sequencial e '
                      "cadeia de hash válida.</p>")
        else:
            items = "".join(f"<li>{esc(p)}</li>" for p in report.problems)
            status = f'<p class="bad">Problemas encontrados:</p><ul>{items}</ul>'
        body = (f'<h1>Integridade das marcações</h1><div class="card">{status}</div>'
                '<p class="muted">Cada marcação guarda o hash da anterior; qualquer alteração '
                "ou exclusão feita por fora do sistema aparece aqui.</p>")
        return page("Integridade", body, "/integridade", csrf)

    def employee_action(self, action: str, form: dict[str, str]) -> str:
        code = form.get("matricula", "").strip()
        try:
            with self.storage() as s:
                if action == "novo":
                    s.add_employee_with_pin(code, form.get("nome", ""), form.get("pin", ""))
                    return "ok=" + quote(f"{form.get('nome', '').strip()} cadastrado(a) com PIN.")
                if action == "pin":
                    s.set_pin(code, form.get("pin", ""))
                    return "ok=" + quote(f"PIN da matrícula {code} definido.")
                if action == "remover":
                    if not s.remove_employee(code):
                        return "erro=" + quote(f"Matrícula {code} não encontrada.")
                    return "ok=" + quote(f"Matrícula {code} removida (rosto e PIN apagados).")
        except StorageError as exc:
            return "erro=" + quote(str(exc))
        return "erro=" + quote("Ação desconhecida.")

    def csv_export(self, kind: str, params: dict[str, str]) -> tuple[str, str]:
        start, end = parse_period(params)
        code = params.get("matricula") or None
        with self.storage() as s:
            punches = s.punches.list(start=start, end=end, employee_code=code)
        name = f"{kind}-{start}-a-{end}.csv"
        if kind == "espelho":
            return name, timesheet_csv(build(punches))
        return name, punches_csv(punches)


def make_handler(panel: Panel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CheckPoint-RH"

        # ---- utilitários

        def _token(self) -> str | None:
            jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
            morsel = jar.get(SESSION_COOKIE)
            return morsel.value if morsel else None

        def _send(self, status: int, ctype: str, body: str | bytes,
                  headers: dict[str, str] | None = None) -> None:
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            if ctype.startswith("text/"):
                ctype += "; charset=utf-8"
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location: str, headers: dict[str, str] | None = None) -> None:
            self._send(303, "text/plain", "", {"Location": location, **(headers or {})})

        def _form(self) -> dict[str, str]:
            length = min(int(self.headers.get("Content-Length") or 0), 64 * 1024)
            raw = self.rfile.read(length).decode("utf-8", "replace")
            return {k: v[0] for k, v in parse_qs(raw).items()}

        # ---- rotas

        def do_GET(self) -> None:  # nome exigido pelo http.server
            url = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/entrar":
                self._send(*panel.login_page()[:3])
                return
            csrf = panel.sessions.csrf_for(self._token())
            if csrf is None:
                self._redirect("/entrar")
                return
            pages = {"/marcacoes": panel.punches_page, "/espelho": panel.timesheet_page,
                     "/funcionarios": panel.employees_page,
                     "/integridade": panel.integrity_page}
            if url.path == "/":
                self._redirect("/marcacoes")
            elif url.path in pages:
                self._send(200, "text/html", pages[url.path](params, csrf))
            elif url.path in ("/csv/marcacoes", "/csv/espelho"):
                name, content = panel.csv_export(url.path.rsplit("/", 1)[1], params)
                self._send(200, "text/csv", content,
                           {"Content-Disposition": f'attachment; filename="{name}"'})
            else:
                self._send(404, "text/plain", "Página não encontrada")

        def do_POST(self) -> None:  # nome exigido pelo http.server
            path = urlparse(self.path).path
            form = self._form()
            if path == "/entrar":
                status, ctype, body, headers = panel.login(self.client_address[0], form)
                self._send(status, ctype, body, headers)
                return
            token = self._token()
            csrf = panel.sessions.csrf_for(token)
            if csrf is None:
                self._redirect("/entrar")
                return
            if not secrets.compare_digest(form.get("csrf", ""), csrf):
                self._send(403, "text/plain", "Requisição recusada (token inválido).")
                return
            if path == "/sair":
                panel.sessions.drop(token)
                self._redirect("/entrar", {"Set-Cookie": f"{SESSION_COOKIE}=; Max-Age=0; Path=/"})
            elif path.startswith("/funcionarios/"):
                result = panel.employee_action(path.rsplit("/", 1)[1], form)
                self._redirect(f"/funcionarios?{result}")
            else:
                self._send(404, "text/plain", "Página não encontrada")

        def log_message(self, fmt: str, *args: object) -> None:
            log.debug("%s - %s", self.address_string(), fmt % args)

    return Handler


def start_panel(panel: Panel, port: int, bind: str = "0.0.0.0") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((bind, port), make_handler(panel))
    threading.Thread(target=server.serve_forever, name="painel", daemon=True).start()
    return server
