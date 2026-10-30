"""Carrega a configuração do totem a partir de um arquivo TOML."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # Ubuntu 22.04 vem com Python 3.10
    import tomli as tomllib

DEFAULT_CONFIG_FILE = Path("config.toml")


class ConfigError(ValueError):
    """Configuração inválida ou ilegível."""


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/checkpoint.db"


@dataclass(frozen=True)
class ModelsConfig:
    dir: Path = Path("models")


@dataclass(frozen=True)
class RecognitionConfig:
    match_threshold: float = 0.363
    detect_score: float = 0.8
    min_face_size: int = 80
    enroll_samples: int = 7


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    dir: Path = Path("logs")


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_section(section: Any, values: dict[str, Any], name: str) -> Any:
    known = {f.name: f.type for f in fields(section)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(f"[{name}] chaves desconhecidas: {', '.join(sorted(unknown))}")

    converted: dict[str, Any] = {}
    for key, value in values.items():
        current = getattr(section, key)
        try:
            if isinstance(current, Path):
                converted[key] = Path(value)
            elif isinstance(current, bool):
                converted[key] = bool(value)
            else:
                converted[key] = type(current)(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"[{name}] valor inválido para '{key}': {value!r}") from exc
    return replace(section, **converted)


def _validate(config: Config) -> None:
    cam = config.camera
    if cam.index < 0:
        raise ConfigError("[camera] index não pode ser negativo")
    if cam.width <= 0 or cam.height <= 0 or cam.fps <= 0:
        raise ConfigError("[camera] width, height e fps devem ser positivos")
    rec = config.recognition
    if not 0.0 < rec.match_threshold < 1.0:
        raise ConfigError("[recognition] match_threshold deve estar entre 0 e 1")
    if not 0.0 < rec.detect_score < 1.0:
        raise ConfigError("[recognition] detect_score deve estar entre 0 e 1")
    if rec.min_face_size <= 0:
        raise ConfigError("[recognition] min_face_size deve ser positivo")
    if not 1 <= rec.enroll_samples <= 30:
        raise ConfigError("[recognition] enroll_samples deve estar entre 1 e 30")
    if config.logging.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("[logging] level deve ser DEBUG, INFO, WARNING ou ERROR")


def load_config(path: Path | None = None) -> Config:
    """Lê o arquivo de configuração; sem arquivo, usa os valores padrão.

    Um caminho passado explicitamente precisa existir. O caminho padrão
    (config.toml na pasta atual) é opcional.
    """
    explicit = path is not None
    path = path or DEFAULT_CONFIG_FILE
    config = Config()

    if not path.exists():
        if explicit:
            raise ConfigError(f"arquivo de configuração não encontrado: {path}")
        return config

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"erro ao ler {path}: {exc}") from exc

    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ConfigError(f"seções desconhecidas: {', '.join(sorted(unknown))}")

    sections = {
        name: _merge_section(getattr(config, name), values, name)
        for name, values in raw.items()
    }
    config = replace(config, **sections)
    _validate(config)
    return config
