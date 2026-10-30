from checkpoint.cli import main


def test_info_is_default_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "CheckPoint" in out
    assert "sqlite:///data/checkpoint.db" in out
    assert (tmp_path / "logs" / "checkpoint.log").exists()


def test_bad_config_returns_2(tmp_path, capsys):
    bad = tmp_path / "config.toml"
    bad.write_text("[camera]\nindex = -1\n", encoding="utf-8")
    assert main(["-c", str(bad), "info"]) == 2
    assert "Erro de configuração" in capsys.readouterr().err


def test_camera_missing_device_returns_1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[camera]\nindex = 9\n", encoding="utf-8")
    assert main(["camera", "--frames", "1"]) == 1
