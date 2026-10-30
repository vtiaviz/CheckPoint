"""Carrega a configuração do totem a partir de um arquivo TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

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
class RecognitionConfig:
    match_threshold: float = 0.363


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    dir: Path = Path("logs")


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
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
    if not 0.0 < config.recognition.match_threshold < 1.0:
        raise ConfigError("[recognition] match_threshold deve estar entre 0 e 1")
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
