import hashlib

import pytest

from checkpoint import models
from checkpoint.models import ModelError, ModelFile, ensure_model

CONTENT = b"modelo de teste"
FAKE = ModelFile("fake.onnx", "http://exemplo.invalid/fake.onnx",
                 hashlib.sha256(CONTENT).hexdigest(), "MIT")


def test_valid_file_is_used_without_download(tmp_path, monkeypatch):
    (tmp_path / "fake.onnx").write_bytes(CONTENT)
    monkeypatch.setattr(models, "_download", lambda *a: pytest.fail("não deveria baixar"))
    assert ensure_model(FAKE, tmp_path) == tmp_path / "fake.onnx"


def test_missing_file_without_download_is_error(tmp_path):
    with pytest.raises(ModelError, match="modelos"):
        ensure_model(FAKE, tmp_path, download=False)


def test_corrupt_file_is_downloaded_again(tmp_path, monkeypatch):
    (tmp_path / "fake.onnx").write_bytes(b"corrompido")
    calls = []

    def fake_download(model, target):
        calls.append(model.filename)
        target.write_bytes(CONTENT)

    monkeypatch.setattr(models, "_download", fake_download)
    ensure_model(FAKE, tmp_path)
    assert calls == ["fake.onnx"]


def test_download_with_wrong_hash_is_rejected(tmp_path, monkeypatch):
    class Response:
        def __init__(self):
            self.chunks = [b"conteudo errado", b""]

        def read(self, _size):
            return self.chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: Response())
    with pytest.raises(ModelError, match="hash inesperado"):
        ensure_model(FAKE, tmp_path)
    assert list(tmp_path.iterdir()) == []
