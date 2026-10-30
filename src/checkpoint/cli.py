"""Linha de comando do CheckPoint."""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path

from checkpoint import __version__
from checkpoint.config import ConfigError, load_config

log = logging.getLogger("checkpoint")


def _setup_logging(level: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "checkpoint.log", encoding="utf-8"),
        ],
    )


def cmd_info(args: argparse.Namespace) -> int:
    cfg = args.config
    print(f"CheckPoint {__version__}")
    print(f"Python {platform.python_version()} em {platform.system()} {platform.release()}")
    try:
        import cv2

        print(f"OpenCV {cv2.__version__}")
    except ImportError:
        print("OpenCV não instalado (pip install -e .)")
    print(f"Câmera: índice {cfg.camera.index}, {cfg.camera.width}x{cfg.camera.height} "
          f"a {cfg.camera.fps} fps")
    print(f"Banco: {cfg.storage.database_url}")
    return 0


def cmd_camera(args: argparse.Namespace) -> int:
    from checkpoint.camera import CameraError, diagnose

    try:
        report = diagnose(args.config.camera, frames=args.frames, preview=args.preview)
    except CameraError as exc:
        log.error("%s", exc)
        return 1

    print(f"Backend: {report.backend}")
    print(f"Resolução entregue: {report.width}x{report.height}")
    print(f"FPS medido: {report.measured_fps:.1f} (pedido: {report.requested_fps})")
    print(f"Brilho médio: {report.mean_brightness:.0f} / 255")
    for warning in report.warnings:
        print(f"AVISO: {warning}")
    if not report.warnings:
        print("Câmera OK para o protótipo.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="checkpoint", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-c", "--config", type=Path, default=None,
                        help="arquivo de configuração (padrão: config.toml, se existir)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="mostra versão, ambiente e configuração").set_defaults(
        func=cmd_info)

    camera = sub.add_parser("camera", help="testa a webcam: resolução, FPS e iluminação")
    camera.add_argument("--frames", type=int, default=90, help="frames a capturar (padrão: 90)")
    camera.add_argument("--preview", action="store_true", help="mostra a imagem durante o teste")
    camera.set_defaults(func=cmd_camera)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.config = load_config(args.config)
    except ConfigError as exc:
        print(f"Erro de configuração: {exc}", file=sys.stderr)
        return 2

    _setup_logging(args.config.logging.level, args.config.logging.dir)
    if not hasattr(args, "func"):
        return cmd_info(args)
    return args.func(args)
