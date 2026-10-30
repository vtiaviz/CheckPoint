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


def cmd_models(args: argparse.Namespace) -> int:
    from checkpoint.models import ALL_MODELS, ModelError, ensure_model

    for model in ALL_MODELS:
        try:
            path = ensure_model(model, args.config.models.dir)
        except ModelError as exc:
            log.error("%s", exc)
            return 1
        print(f"OK  {path}  ({model.license})")
    return 0


def _engine_and_storage(cfg):
    from checkpoint.app import build_engine
    from checkpoint.storage import Storage

    return build_engine(cfg), Storage.from_url(cfg.storage.database_url)


def cmd_enroll(args: argparse.Namespace) -> int:
    from checkpoint.app import EnrollError, camera_frames, enroll
    from checkpoint.camera import CameraError
    from checkpoint.storage import StorageError

    cfg = args.config
    engine, storage = _engine_and_storage(cfg)
    with storage:
        existing = storage.find_employee(args.matricula)
        if existing and not args.adicionar:
            log.error("matrícula %s já cadastrada (use --adicionar para incluir amostras)",
                      args.matricula)
            return 1
        if args.adicionar and not existing:
            log.error("matrícula %s não encontrada", args.matricula)
            return 1
        name = existing.name if existing else args.nome
        if not name:
            log.error("informe o nome do funcionário")
            return 1

        print(f"Cadastrando {name}. Olhe para a câmera e mexa levemente a cabeça.")
        try:
            samples = enroll(cfg, engine, storage.load_gallery(), camera_frames(cfg),
                             preview=not args.sem_preview,
                             same_employee_id=existing.id if existing else None)
            if existing:
                total = storage.add_samples(args.matricula, samples)
            else:
                storage.add_employee(args.matricula, name, samples)
                total = len(samples)
        except (EnrollError, CameraError, StorageError) as exc:
            log.error("%s", exc)
            return 1
    print(f"{name} cadastrado com {total} amostras.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from checkpoint.storage import Storage

    with Storage.from_url(args.config.storage.database_url) as storage:
        employees = storage.list_employees()
    if not employees:
        print("Nenhum funcionário cadastrado.")
    for e in employees:
        print(f"{e.code:>10}  {e.name}  ({e.samples} amostras)")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    from checkpoint.storage import Storage

    with Storage.from_url(args.config.storage.database_url) as storage:
        removed = storage.remove_employee(args.matricula)
    if not removed:
        log.error("matrícula %s não encontrada", args.matricula)
        return 1
    print(f"Matrícula {args.matricula} removida, com seus vetores faciais.")
    return 0


def cmd_recognize(args: argparse.Namespace) -> int:
    from checkpoint.app import camera_frames, recognize
    from checkpoint.camera import CameraError

    cfg = args.config
    engine, storage = _engine_and_storage(cfg)
    with storage:
        gallery = storage.load_gallery()
    if not len(gallery):
        log.error("nenhum funcionário cadastrado; use o comando cadastrar")
        return 1

    def announce(match, best):
        if match:
            print(f"Reconhecido: {match.name} (similaridade {match.score:.2f})")
        elif best:
            print(f"Rosto não reconhecido (mais parecido: {best.name}, "
                  f"similaridade {best.score:.2f}, limiar {cfg.recognition.match_threshold})")
        else:
            print("Rosto não reconhecido")

    print(f"{len(gallery)} funcionário(s) carregado(s). Pressione Q na janela para sair.")
    try:
        recognize(cfg, engine, gallery, camera_frames(cfg), preview=not args.sem_preview,
                  on_change=announce)
    except CameraError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    import itertools

    import numpy as np

    from checkpoint.app import benchmark, build_engine, camera_frames
    from checkpoint.camera import CameraError
    from checkpoint.face import normalize
    from checkpoint.gallery import Gallery

    cfg = args.config
    engine = build_engine(cfg)

    # Galeria sintética do tamanho alvo (50 pessoas x amostras do cadastro) para medir a busca.
    rng = np.random.default_rng(0)
    gallery = Gallery()
    for i in range(args.pessoas):
        gallery.add(i, f"p{i}", [normalize(rng.standard_normal(128))
                                 for _ in range(cfg.recognition.enroll_samples)])

    if args.imagem:
        import cv2

        image = cv2.imread(str(args.imagem))
        if image is None:
            log.error("não foi possível ler a imagem %s", args.imagem)
            return 1
        frames = itertools.repeat(image)
        print(f"Imagem {args.imagem} ({image.shape[1]}x{image.shape[0]}), {args.frames} repetições")
    else:
        frames = camera_frames(cfg)
        print(f"Câmera {cfg.camera.width}x{cfg.camera.height}, {args.frames} frames. "
              "Fique diante da câmera.")
    try:
        result = benchmark(engine, gallery, frames, args.frames)
    except CameraError as exc:
        log.error("%s", exc)
        return 1
    print(f"Galeria: {args.pessoas} pessoas")
    print(result.report())
    return 0


def _liveness_parts(cfg):
    from checkpoint.liveness import LandmarkSensor, PassiveAntiSpoof
    from checkpoint.models import LANDMARKER, ensure_model

    return LandmarkSensor(ensure_model(LANDMARKER, cfg.models.dir)), PassiveAntiSpoof()


def cmd_punch(args: argparse.Namespace) -> int:
    from checkpoint.app import (
        GREEN,
        RED,
        camera_frames,
        verify_liveness,
        wait_for_employee,
        wait_until_gone,
    )
    from checkpoint.camera import CameraError

    cfg = args.config
    preview = not args.sem_preview
    engine, storage = _engine_and_storage(cfg)
    with storage:
        gallery = storage.load_gallery()
    if not len(gallery):
        log.error("nenhum funcionário cadastrado; use o comando cadastrar")
        return 1
    sensor, antispoof = _liveness_parts(cfg)
    print("Simulação de batida (nada é registrado ainda). Pressione Q na janela para sair.")
    if not cfg.liveness.enabled:
        print("AVISO: prova de vida desligada em [liveness] enabled.")

    try:
        frames = camera_frames(cfg)
        while True:
            employee = wait_for_employee(cfg, engine, gallery, frames, preview)
            if employee is None:
                break
            if cfg.liveness.enabled:
                result = verify_liveness(cfg, engine, sensor, antispoof, gallery, employee,
                                         frames, preview)
            else:
                from checkpoint.liveness import LivenessResult

                result = LivenessResult(True)
            if result.reason == "cancelado":
                break
            gestures = ", ".join(c.value.lower() for c in result.challenges) or "-"
            if result.passed:
                print(f"APROVADO: {employee.name} | similaridade {employee.score:.2f} | "
                      f"rosto real {result.real_score:.2f} | {result.elapsed_s:.1f}s | "
                      f"desafio: {gestures}")
                text, color = f"Ponto OK: {employee.name}", GREEN
            else:
                print(f"RECUSADO: {employee.name} | {result.reason} | "
                      f"rosto real {result.real_score:.2f} | {result.elapsed_s:.1f}s | "
                      f"desafio: {gestures}")
                text, color = "Recusado - tente de novo", RED
            if not wait_until_gone(engine, frames, text, color, preview):
                break
    except CameraError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
        if preview:
            import cv2

            cv2.destroyAllWindows()
    return 0


def cmd_liveness_diag(args: argparse.Namespace) -> int:
    import time

    from checkpoint.app import GREEN, RED, YELLOW, camera_frames, check_face, show_mirrored
    from checkpoint.camera import CameraError

    cfg = args.config
    from checkpoint.app import build_engine

    engine = build_engine(cfg)
    sensor, antispoof = _liveness_parts(cfg)
    threshold = cfg.liveness.real_threshold
    print("Diagnóstico da prova de vida. Mostre um rosto real, uma foto e o celular.")
    print(f"'rosto real' acima de {threshold} conta como rosto verdadeiro. Q para sair.")
    last_print = 0.0
    try:
        for frame in camera_frames(cfg):
            face, problem = check_face(engine.detect(frame), cfg.recognition.min_face_size)
            if face is None or problem:
                if not show_mirrored(frame, face, YELLOW, problem or "Nenhum rosto"):
                    break
                continue
            now = time.monotonic()
            signals = sensor.measure(frame, int(now * 1000))
            real = antispoof.score(frame, face)
            blink = f"{signals.blink:.2f}" if signals else "-"
            yaw = f"{signals.yaw:+.0f}" if signals else "-"
            line = f"rosto real {real:.2f} | piscada {blink} | giro {yaw} graus"
            if now - last_print >= 0.5:
                print(line)
                last_print = now
            ok = real >= threshold
            if not show_mirrored(frame, face, GREEN if ok else RED,
                                 "REAL" if ok else "SUSPEITO", line):
                break
    except CameraError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
        import cv2

        cv2.destroyAllWindows()
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

    sub.add_parser("modelos", help="baixa e verifica os modelos YuNet e SFace").set_defaults(
        func=cmd_models)

    enroll = sub.add_parser("cadastrar", help="cadastra o rosto de um funcionário")
    enroll.add_argument("matricula", help="matrícula ou código do funcionário")
    enroll.add_argument("nome", nargs="?",
                        help="nome do funcionário (entre aspas se tiver espaços)")
    enroll.add_argument("--adicionar", action="store_true",
                        help="acrescenta amostras a quem já está cadastrado (ex.: outra luz)")
    enroll.add_argument("--sem-preview", action="store_true", help="não abre a janela de vídeo")
    enroll.set_defaults(func=cmd_enroll)

    sub.add_parser("listar", help="lista os funcionários cadastrados").set_defaults(
        func=cmd_list)

    remove = sub.add_parser("remover", help="remove um funcionário e seus vetores faciais")
    remove.add_argument("matricula")
    remove.set_defaults(func=cmd_remove)

    recognize = sub.add_parser("reconhecer", help="identifica ao vivo quem está na câmera")
    recognize.add_argument("--sem-preview", action="store_true",
                           help="não abre a janela; só imprime no terminal")
    recognize.set_defaults(func=cmd_recognize)

    punch = sub.add_parser("ponto", help="simula a batida: reconhecimento + prova de vida")
    punch.add_argument("--sem-preview", action="store_true",
                       help="não abre a janela (o desafio não aparece na tela)")
    punch.set_defaults(func=cmd_punch)

    sub.add_parser("vivacidade",
                   help="mostra ao vivo os sinais da prova de vida, para calibrar").set_defaults(
        func=cmd_liveness_diag)

    bench = sub.add_parser("benchmark", help="mede o tempo de cada etapa do reconhecimento")
    bench.add_argument("--frames", type=int, default=100, help="frames a medir (padrão: 100)")
    bench.add_argument("--pessoas", type=int, default=50,
                       help="tamanho da galeria sintética (padrão: 50)")
    bench.add_argument("--imagem", type=Path, help="usa uma foto em vez da câmera")
    bench.set_defaults(func=cmd_benchmark)
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

    from checkpoint.models import ModelError

    try:
        return args.func(args)
    except ModelError as exc:
        log.error("%s", exc)
        return 1
