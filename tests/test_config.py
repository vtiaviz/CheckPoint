from pathlib import Path

import pytest

from checkpoint.config import Config, ConfigError, load_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_default_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config() == Config()


def test_explicit_missing_file_is_error(tmp_path):
    with pytest.raises(ConfigError, match="não encontrado"):
        load_config(tmp_path / "nao_existe.toml")


def test_overrides_and_types(tmp_path):
    cfg = load_config(write(tmp_path, """
[camera]
index = 1
width = 1280
height = 720

[storage]
data_dir = "dados"
"""))
    assert cfg.camera.index == 1
    assert (cfg.camera.width, cfg.camera.height) == (1280, 720)
    assert cfg.camera.fps == 30
    assert cfg.storage.data_dir == Path("dados")


def test_example_config_is_valid():
    example = Path(__file__).parents[1] / "config.example.toml"
    assert load_config(example) == Config()


@pytest.mark.parametrize("text, message", [
    ("[camera]\nzoom = 2\n", "chaves desconhecidas"),
    ("[webcam]\nindex = 0\n", "seções desconhecidas"),
    ("[camera]\nwidth = 'grande'\n", "valor inválido"),
    ("[camera]\nindex = -1\n", "negativo"),
    ("[recognition]\nmatch_threshold = 1.5\n", "entre 0 e 1"),
    ("[logging]\nlevel = 'VERBOSE'\n", "level"),
    ("[camera\n", "erro ao ler"),
])
def test_invalid_config(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text))
