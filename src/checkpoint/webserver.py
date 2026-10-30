"""Servidor HTTP na rede local para o funcionário abrir o comprovante pelo QR code.

Rotas:
    /c/<nsr>/<token>      página simples para celular com os dados e o botão do PDF
    /c/<nsr>/<token>.pdf  o comprovante em PDF

Sem token válido a resposta é 404, igual a uma marcação inexistente.
"""

from __future__ import annotations

import html
import logging
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from checkpoint.punches import Punch
from checkpoint.receipt import DISCLAIMER, receipt_fields, receipt_url, render_pdf, token_ok
from checkpoint.storage import Storage

log = logging.getLogger(__name__)

ROUTE = re.compile(r"^/c/(\d{1,12})/([0-9a-f]{16})(\.pdf)?$")


@dataclass(frozen=True)
class ReceiptSite:
    database_url: str
    secret: bytes
    company: str
    cnpj: str
    location: str
    base_url: str

    def find(self, nsr: int, token: str) -> Punch | None:
        # Uma conexão por requisição: o sqlite3 não compartilha conexões entre threads.
        with Storage.from_url(self.database_url) as storage:
            punch = storage.punches.get(nsr)
        if punch is None or not token_ok(self.secret, punch, token):
            return None
        return punch

    def pdf(self, punch: Punch) -> bytes:
        return render_pdf(punch, self.company, self.cnpj, self.location,
                          receipt_url(self.base_url, self.secret, punch))

    def page(self, punch: Punch, pdf_href: str) -> str:
        rows = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
            for k, v in receipt_fields(punch, self.company, self.cnpj, self.location)
        )
        return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comprovante NSR {punch.nsr}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1a1a1a; background: #fafafa; }}
 h1 {{ font-size: 1.2rem; }} table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ text-align: left; padding: 6px 4px; border-bottom: 1px solid #ddd; }}
 th {{ width: 40%; color: #555; font-weight: 600; }}
 code {{ word-break: break-all; font-size: .75rem; }}
 a.btn {{ display: block; margin: 20px 0; padding: 14px; text-align: center; background: #1f6f43;
         color: #fff; border-radius: 8px; text-decoration: none; font-weight: 600; }}
 small {{ color: #666; }}
</style></head><body>
<h1>Comprovante de Registro de Ponto</h1>
<table>{rows}</table>
<p><small>Hash SHA-256:</small><br><code>{punch.hash}</code></p>
<a class="btn" href="{html.escape(pdf_href)}" download>Baixar PDF</a>
<p><small>{html.escape(DISCLAIMER)}</small></p>
</body></html>"""


def make_handler(site: ReceiptSite) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CheckPoint"

        def do_GET(self) -> None:  # nome exigido pelo http.server
            path = self.path.split("?", 1)[0]
            if path == "/":
                # Página de teste: se abre no celular, a rede e o firewall estão OK.
                self._send(200, "text/plain; charset=utf-8",
                           b"CheckPoint: servidor de comprovantes no ar.")
                return
            match = ROUTE.match(path)
            punch = site.find(int(match[1]), match[2]) if match else None
            if punch is None:
                self._send(404, "text/plain; charset=utf-8", "Comprovante não encontrado".encode())
                return
            if match[3]:
                headers = {"Content-Disposition":
                           f'attachment; filename="comprovante-nsr-{punch.nsr}.pdf"'}
                self._send(200, "application/pdf", site.pdf(punch), headers)
            else:
                body = site.page(punch, f"{self.path.split('?', 1)[0]}.pdf").encode()
                self._send(200, "text/html; charset=utf-8", body)

        def _send(self, status: int, ctype: str, body: bytes,
                  headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            log.debug("%s - %s", self.address_string(), fmt % args)

    return Handler


def start_server(site: ReceiptSite, port: int, bind: str = "0.0.0.0") -> ThreadingHTTPServer:
    """Sobe o servidor numa thread em segundo plano; chame shutdown() para parar."""
    server = ThreadingHTTPServer((bind, port), make_handler(site))
    threading.Thread(target=server.serve_forever, name="comprovantes", daemon=True).start()
    return server
