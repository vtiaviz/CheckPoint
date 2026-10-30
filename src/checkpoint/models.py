"""Download e verificação dos modelos ONNX do OpenCV Zoo."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"


@dataclass(frozen=True)
class ModelFile:
    filename: str
    url: str
    sha256: str
    license: str


# Os dois modelos permitem uso comercial (ver README).
DETECTOR = ModelFile(
    filename="face_detection_yunet_2023mar.onnx",
    url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    license="MIT",
)
RECOGNIZER = ModelFile(
    filename="face_recognition_sface_2021dec.onnx",
    url=f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    license="Apache-2.0",
)
LANDMARKER = ModelFile(
    filename="face_landmarker.task",
    url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task",
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    license="Apache-2.0",
)
ALL_MODELS = (DETECTOR, RECOGNIZER, LANDMARKER)


class ModelError(RuntimeError):
    """Modelo ausente, corrompido ou impossível de baixar."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(model: ModelFile, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    log.info("Baixando %s ...", model.filename)
    try:
        with urllib.request.urlopen(model.url, timeout=60) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ModelError(f"falha ao baixar {model.filename}: {exc}") from exc

    actual = sha256_of(tmp)
    if actual != model.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelError(f"{model.filename} baixado com hash inesperado ({actual})")
    tmp.replace(target)


def ensure_model(model: ModelFile, models_dir: Path, download: bool = True) -> Path:
    """Devolve o caminho do modelo, baixando-o se faltar. Arquivo corrompido é baixado de novo."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / model.filename
    if path.exists() and sha256_of(path) == model.sha256:
        return path
    if not download:
        raise ModelError(
            f"modelo {model.filename} ausente ou corrompido em {models_dir}. "
            "Rode: python -m checkpoint modelos"
        )
    _download(model, path)
    return path
