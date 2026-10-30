import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from checkpoint.face import normalize
from checkpoint.punches import FACE
from checkpoint.receipt import (
    load_secret,
    qr_image,
    qr_matrix,
    receipt_fields,
    receipt_url,
    render_pdf,
    token_for,
    token_ok,
)
from checkpoint.storage import Storage
from checkpoint.webserver import ReceiptSite, start_server

BRT = timezone(timedelta(hours=-3))


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{(tmp_path / 'data' / 'checkpoint.db').as_posix()}"
    with Storage.from_url(url) as s:
        s.add_employee("001", "João", [normalize(np.ones(128))])
        punch = s.punches.register("001", "João", FACE, 0.9, 0.98,
                                   now=datetime(2026, 10, 30, 8, 1, 23, tzinfo=BRT))
    return url, punch, tmp_path / "data"


def test_secret_is_created_once(tmp_path):
    first = load_secret(tmp_path)
    assert len(first) == 32 and load_secret(tmp_path) == first


def test_token_depends_on_secret_and_punch(db):
    _, punch, data = db
    secret = load_secret(data)
    token = token_for(secret, punch)
    assert len(token) == 16 and token_ok(secret, punch, token)
    assert not token_ok(b"x" * 32, punch, token)


def test_fields_and_pdf(db):
    _, punch, _ = db
    fields = dict(receipt_fields(punch, "ACME", "", "Portaria"))
    assert fields["Horário"] == "08:01:23 (UTC-03:00)"
    assert fields["NSR"] == "000000001" and fields["CNPJ"] == "não informado"
    assert fields["Tipo"] == "Entrada"
    pdf = render_pdf(punch, "ACME", "", "Portaria", "http://x/c/1/abc")
    assert pdf.startswith(b"%PDF-") and len(pdf) > 1000


def test_qr_matrix_and_image():
    matrix = qr_matrix("http://192.168.0.10:8080/c/1/0123456789abcdef")
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert matrix[:4].sum() == 0  # margem branca
    image = qr_image("abc", size=200)
    assert image.shape == (200, 200, 3)


def test_server_serves_page_and_pdf_only_with_valid_token(db):
    url, punch, data = db
    site = ReceiptSite(url, load_secret(data), "ACME", "", "Portaria", "http://127.0.0.1:0")
    server = start_server(site, 0, bind="127.0.0.1")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"no ar" in urllib.request.urlopen(base + "/").read()
        link = receipt_url(base, site.secret, punch)
        page = urllib.request.urlopen(link).read().decode()
        assert "João" in page and punch.hash in page and ".pdf" in page
        response = urllib.request.urlopen(link + ".pdf")
        assert response.headers["Content-Type"] == "application/pdf"
        assert response.read().startswith(b"%PDF-")
        for bad in (f"{base}/c/1/{'0' * 16}", f"{base}/c/2/{token_for(site.secret, punch)}",
                    f"{base}/outra-coisa"):
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(bad)
            assert err.value.code == 404
    finally:
        server.shutdown()
