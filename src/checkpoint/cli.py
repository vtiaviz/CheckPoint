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
    from checkpoint.liveness import describe_yaw

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
            yaw = f"{signals.yaw:+.0f} graus ({describe_yaw(signals.yaw)})" if signals else "-"
            line = f"rosto real {real:.2f} | piscada {blink} | giro {yaw}"
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


def cmd_totem(args: argparse.Namespace) -> int:
    from checkpoint.app import camera_frames
    from checkpoint.camera import CameraError
    from checkpoint.totem import run_totem

    cfg = args.config
    engine, storage = _engine_and_storage(cfg)
    sensor, antispoof = _liveness_parts(cfg)
    with storage:
        if not storage.list_employees():
            log.error("nenhum funcionário cadastrado; use o comando cadastrar")
            return 1
        report = storage.punches.verify()
        if not report.ok:
            log.warning("integridade das marcações com problemas; rode: python -m checkpoint "
                        "verificar")
        server, url_for = _start_receipts(cfg)
        print("Totem ativo. Pressione Q na janela para encerrar.")
        try:
            run_totem(cfg, engine, sensor, antispoof, storage, camera_frames(cfg),
                      receipt_url_for=url_for)
        except CameraError as exc:
            log.error("%s", exc)
            return 1
        except KeyboardInterrupt:
            pass
        finally:
            sensor.close()
            if server is not None:
                server.shutdown()
            import cv2

            cv2.destroyAllWindows()
    return 0


def _receipt_site(cfg):
    from checkpoint.receipt import load_secret, local_ip
    from checkpoint.webserver import ReceiptSite

    host = cfg.receipt.host or local_ip()
    return ReceiptSite(
        database_url=cfg.storage.database_url,
        secret=load_secret(cfg.storage.data_dir),
        company=cfg.company.name,
        cnpj=cfg.company.cnpj,
        location=cfg.company.location,
        base_url=f"http://{host}:{cfg.receipt.port}",
    )


def _start_receipts(cfg):
    """Sobe o servidor de comprovantes; sem ele o totem funciona, só não mostra o QR."""
    if not cfg.receipt.enabled:
        return None, None
    from checkpoint.receipt import receipt_url
    from checkpoint.webserver import start_server

    site = _receipt_site(cfg)
    try:
        server = start_server(site, cfg.receipt.port)
    except OSError as exc:
        log.warning("comprovantes desligados: porta %s indisponível (%s)", cfg.receipt.port, exc)
        return None, None
    print(f"Comprovantes em {site.base_url} (o celular precisa estar na mesma rede)")
    return server, lambda punch: receipt_url(site.base_url, site.secret, punch)


def cmd_receipt(args: argparse.Namespace) -> int:
    from checkpoint.receipt import receipt_url
    from checkpoint.storage import Storage

    cfg = args.config
    with Storage.from_url(cfg.storage.database_url) as storage:
        punch = storage.punches.get(args.nsr)
    if punch is None:
        log.error("NSR %s não encontrado", args.nsr)
        return 1
    site = _receipt_site(cfg)
    out = args.saida or Path(f"comprovante-nsr-{punch.nsr}.pdf")
    out.write_bytes(site.pdf(punch))
    print(f"PDF salvo em {out}")
    print(f"Link (com o totem aberto): {receipt_url(site.base_url, site.secret, punch)}")
    return 0


def cmd_panel_password(args: argparse.Namespace) -> int:
    import getpass

    from checkpoint.panel import PanelError, set_password
    from checkpoint.storage import Storage

    password = getpass.getpass("Nova senha do painel do RH (mínimo 8 caracteres): ")
    if password != getpass.getpass("Repita a senha: "):
        log.error("as senhas não conferem")
        return 1
    with Storage.from_url(args.config.storage.database_url) as storage:
        try:
            set_password(storage, password)
        except PanelError as exc:
            log.error("%s", exc)
            return 1
    print("Senha do painel definida.")
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    import time

    from checkpoint.panel import PASSWORD_KEY, Panel, start_panel
    from checkpoint.receipt import local_ip
    from checkpoint.storage import Storage

    cfg = args.config
    with Storage.from_url(cfg.storage.database_url) as storage:
        if storage.get_setting(PASSWORD_KEY) is None:
            log.error("defina a senha antes: python -m checkpoint senha-painel")
            return 1
    try:
        server = start_panel(Panel(cfg.storage.database_url), cfg.panel.port, cfg.panel.bind)
    except OSError as exc:
        log.error("não foi possível abrir a porta %s: %s", cfg.panel.port, exc)
        return 1
    host = local_ip() if cfg.panel.bind == "0.0.0.0" else cfg.panel.bind
    print(f"Painel do RH em http://{host}:{cfg.panel.port} (Ctrl+C para encerrar)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


def cmd_pin(args: argparse.Namespace) -> int:
    import getpass

    from checkpoint.storage import Storage, StorageError

    with Storage.from_url(args.config.storage.database_url) as storage:
        employee = storage.find_employee(args.matricula)
        if employee is None:
            log.error("matrícula %s não encontrada", args.matricula)
            return 1
        pin = getpass.getpass(f"Novo PIN para {employee.name} (4 a 8 dígitos): ")
        if pin != getpass.getpass("Repita o PIN: "):
            log.error("os PINs não conferem")
            return 1
        try:
            storage.set_pin(args.matricula, pin)
        except StorageError as exc:
            log.error("%s", exc)
            return 1
    print(f"PIN de {employee.name} definido.")
    return 0


def cmd_punches(args: argparse.Namespace) -> int:
    from datetime import date

    from checkpoint.storage import Storage

    day = None
    if not args.todas:
        try:
            day = date.fromisoformat(args.data) if args.data else date.today()
        except ValueError:
            log.error("data inválida: %s (use AAAA-MM-DD)", args.data)
            return 1
    with Storage.from_url(args.config.storage.database_url) as storage:
        punches = storage.punches.list(day=day, employee_code=args.matricula)
    if not punches:
        print("Nenhuma marcação encontrada.")
    for p in punches:
        method = "PIN" if p.method == "pin" else "face"
        print(f"NSR {p.nsr:>6} | {p.local_date} {p.time_text} | {p.kind_text:<7} | "
              f"{p.employee_code:>8} {p.employee_name} | {method}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from checkpoint.storage import Storage

    with Storage.from_url(args.config.storage.database_url) as storage:
        report = storage.punches.verify()
    if report.ok:
        print(f"OK: {report.total} marcações íntegras, NSR sequencial e cadeia de hash válida.")
        return 0
    print(f"FALHA: {len(report.problems)} problema(s) em {report.total} marcações:")
    for problem in report.problems:
        print(f"  - {problem}")
    return 1


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

    punch = sub.add_parser("ponto", help="simula a batida sem registrar (teste)")
    punch.add_argument("--sem-preview", action="store_true",
                       help="não abre a janela (o desafio não aparece na tela)")
    punch.set_defaults(func=cmd_punch)

    sub.add_parser("totem", help="modo totem: reconhece, confirma e registra o ponto").set_defaults(
        func=cmd_totem)

    pin = sub.add_parser("pin", help="define o PIN do funcionário (alternativa à biometria)")
    pin.add_argument("matricula")
    pin.set_defaults(func=cmd_pin)

    punches = sub.add_parser("marcacoes", help="lista as marcações (padrão: hoje)")
    punches.add_argument("--data", help="dia no formato AAAA-MM-DD")
    punches.add_argument("--matricula", help="só deste funcionário")
    punches.add_argument("--todas", action="store_true", help="todas as datas")
    punches.set_defaults(func=cmd_punches)

    sub.add_parser("verificar", help="confere a integridade das marcações").set_defaults(
        func=cmd_verify)

    sub.add_parser("painel", help="painel web do RH (marcações, espelho, funcionários)"
                   ).set_defaults(func=cmd_panel)
    sub.add_parser("senha-painel", help="define a senha de acesso ao painel do RH"
                   ).set_defaults(func=cmd_panel_password)

    receipt = sub.add_parser("comprovante", help="gera o PDF do comprovante de uma marcação")
    receipt.add_argument("nsr", type=int)
    receipt.add_argument("--saida", type=Path, help="arquivo PDF (padrão: comprovante-nsr-N.pdf)")
    receipt.set_defaults(func=cmd_receipt)

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
