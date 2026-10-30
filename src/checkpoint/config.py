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
class LivenessConfig:
    enabled: bool = True
    challenges: int = 1
    timeout_s: float = 6.0
    real_threshold: float = 0.6
    min_passive_frames: int = 3


@dataclass(frozen=True)
class PunchConfig:
    min_interval_min: float = 2.0
    pin_max_attempts: int = 3
    pin_lockout_min: float = 5.0


@dataclass(frozen=True)
class CompanyConfig:
    name: str = "Empresa (protótipo)"
    cnpj: str = ""
    location: str = "Portaria"


@dataclass(frozen=True)
class ReceiptConfig:
    enabled: bool = True
    port: int = 8080
    host: str = ""  # vazio = IP do computador na rede local, detectado automaticamente
    qr_seconds: float = 10.0


@dataclass(frozen=True)
class PanelConfig:
    port: int = 8081
    bind: str = "0.0.0.0"  # "127.0.0.1" deixa o painel acessível só neste computador


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
    liveness: LivenessConfig = field(default_factory=LivenessConfig)
    punch: PunchConfig = field(default_factory=PunchConfig)
    company: CompanyConfig = field(default_factory=CompanyConfig)
    receipt: ReceiptConfig = field(default_factory=ReceiptConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
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
    live = config.liveness
    if not 1 <= live.challenges <= 2:
        raise ConfigError("[liveness] challenges deve ser 1 ou 2")
    if live.timeout_s <= 0:
        raise ConfigError("[liveness] timeout_s deve ser positivo")
    if not 0.0 < live.real_threshold < 1.0:
        raise ConfigError("[liveness] real_threshold deve estar entre 0 e 1")
    if live.min_passive_frames < 1:
        raise ConfigError("[liveness] min_passive_frames deve ser pelo menos 1")
    punch = config.punch
    if punch.min_interval_min < 0 or punch.pin_lockout_min < 0:
        raise ConfigError("[punch] intervalos não podem ser negativos")
    if punch.pin_max_attempts < 1:
        raise ConfigError("[punch] pin_max_attempts deve ser pelo menos 1")
    if not 1 <= config.receipt.port <= 65535:
        raise ConfigError("[receipt] port deve estar entre 1 e 65535")
    if config.receipt.qr_seconds < 0:
        raise ConfigError("[receipt] qr_seconds não pode ser negativo")
    if not 1 <= config.panel.port <= 65535:
        raise ConfigError("[panel] port deve estar entre 1 e 65535")
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
