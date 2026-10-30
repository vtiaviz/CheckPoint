"""Comprovante da marcação: PDF, link com token e matriz do QR code.

O link leva um token HMAC derivado do NSR e do hash da marcação. Sem a chave secreta do totem
não dá para montar o link de outra marcação, então ninguém na rede baixa o comprovante alheio
trocando o número na URL.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import secrets
import socket
from datetime import datetime
from pathlib import Path

import numpy as np

from checkpoint.punches import PIN, Punch

TOKEN_LEN = 16
SECRET_FILE = "receipt_secret"
DISCLAIMER = ("Protótipo de controle interno. Não é comprovante de REP-P "
              "(Portaria MTP 671/2021) e não tem assinatura digital ICP-Brasil.")


def load_secret(data_dir: Path) -> bytes:
    """Chave do HMAC, criada na primeira execução e guardada só neste computador."""
    path = data_dir / SECRET_FILE
    if path.exists():
        return bytes.fromhex(path.read_text(encoding="ascii").strip())
    data_dir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_text(key.hex(), encoding="ascii")
    try:
        os.chmod(path, 0o600)
    except OSError:  # Windows: permissões ficam com as da pasta
        pass
    return key


def token_for(secret: bytes, punch: Punch) -> str:
    message = f"{punch.nsr}:{punch.hash}".encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()[:TOKEN_LEN]


def token_ok(secret: bytes, punch: Punch, token: str) -> bool:
    return hmac.compare_digest(token_for(secret, punch), token)


def local_ip() -> str:
    """IP deste computador na rede local (sem enviar nada: o UDP só escolhe a rota)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def receipt_url(base_url: str, secret: bytes, punch: Punch) -> str:
    return f"{base_url.rstrip('/')}/c/{punch.nsr}/{token_for(secret, punch)}"


def qr_matrix(text: str) -> np.ndarray:
    """Matriz do QR (True = módulo escuro), já com a margem de 4 módulos."""
    import qrcode

    qr = qrcode.QRCode(border=4, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(text)
    qr.make(fit=True)
    return np.array(qr.get_matrix(), dtype=bool)


def qr_image(text: str, size: int = 220) -> np.ndarray:
    """QR como imagem BGR quadrada, pronta para desenhar sobre o frame da câmera."""
    import cv2

    matrix = qr_matrix(text)
    gray = np.where(matrix, 0, 255).astype(np.uint8)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def receipt_fields(punch: Punch, company: str, cnpj: str, location: str) -> list[tuple[str, str]]:
    moment = datetime.fromisoformat(punch.punched_at)
    offset = moment.strftime("%z")  # ex.: -0300
    rows = [
        ("Empregador", company),
        ("CNPJ", cnpj or "não informado"),
        ("Local", location),
        ("Trabalhador", punch.employee_name),
        ("Matrícula", punch.employee_code),
        ("Data", moment.strftime("%d/%m/%Y")),
        ("Horário", f"{moment.strftime('%H:%M:%S')} (UTC{offset[:3]}:{offset[3:]})"),
        ("Tipo", punch.kind_text),
        ("Identificação", "matrícula e PIN" if punch.method == PIN else "reconhecimento facial"),
        ("NSR", f"{punch.nsr:09d}"),
    ]
    return rows


def render_pdf(punch: Punch, company: str, cnpj: str, location: str, url: str = "") -> bytes:
    """PDF do comprovante em tamanho de bobina (80 mm), legível no celular."""
    from reportlab.graphics import renderPDF
    from reportlab.graphics.barcode.qr import QrCodeWidget
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    width, height = 80 * mm, 150 * mm
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(width, height))
    pdf.setTitle(f"Comprovante NSR {punch.nsr}")
    pdf.setAuthor(company)

    y = height - 12 * mm
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(width / 2, y, "Comprovante de Registro de Ponto")
    y -= 5 * mm
    pdf.drawCentredString(width / 2, y, "do Trabalhador")
    y -= 8 * mm

    for label, value in receipt_fields(punch, company, cnpj, location):
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(6 * mm, y, f"{label}:")
        pdf.setFont("Helvetica", 8)
        pdf.drawString(28 * mm, y, value[:42])
        y -= 5 * mm

    y -= 2 * mm
    pdf.setFont("Helvetica-Bold", 7)
    pdf.drawString(6 * mm, y, "Hash SHA-256 da marcação:")
    pdf.setFont("Courier", 6.5)
    for i in range(0, len(punch.hash), 32):
        y -= 3.5 * mm
        pdf.drawString(6 * mm, y, punch.hash[i:i + 32])

    if url:
        widget = QrCodeWidget(url)
        x0, y0, x1, y1 = widget.getBounds()
        side = 28 * mm
        drawing = Drawing(side, side, transform=[side / (x1 - x0), 0, 0, side / (y1 - y0), 0, 0])
        drawing.add(widget)
        y -= side + 4 * mm
        renderPDF.draw(drawing, pdf, (width - side) / 2, y)

    y -= 6 * mm
    pdf.setFont("Helvetica-Oblique", 6)
    text = pdf.beginText(6 * mm, y)
    for line in _wrap(DISCLAIMER, 60):
        text.textLine(line)
    pdf.drawText(text)
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return [*lines, current] if current else lines
