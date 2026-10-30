# CheckPoint

Protótipo de ponto eletrônico por reconhecimento facial, com processamento 100% local e
bibliotecas de código aberto. Escopo atual: **controle interno**, com **um ponto de coleta**
(totem) e até 50 funcionários. Não é, por ora, um REP-P da Portaria MTP 671/2021.

Detalhes de escopo, arquitetura e milestones em [docs/plano.md](docs/plano.md).

## Requisitos

- Python 3.10 a 3.12 (o MediaPipe, usado a partir do M2, ainda não cobre todas as versões novas)
- Webcam USB (testado com Logitech C270, 720p)
- Windows 10/11 ou Ubuntu LTS

Webcam dentro de máquina virtual costuma falhar ou travar. Para testar, rode direto no Windows;
para validar o Linux, use um pendrive com Ubuntu.

## Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy config.example.toml config.toml   # Linux: cp config.example.toml config.toml
```

## Uso

```bash
python -m checkpoint                     # versão, ambiente e configuração
python -m checkpoint camera              # testa a webcam: resolução, FPS e iluminação
python -m checkpoint camera --preview    # idem, mostrando a imagem (Q para sair)
```

O teste da câmera avisa quando a imagem está escura ou o FPS está baixo. Iluminação frontal
no ponto de captura importa mais que a resolução da webcam.

## Testes

```bash
pytest
ruff check .
```

## Licenças de terceiros

Só são usados componentes que permitem uso comercial: OpenCV e modelos YuNet/SFace do OpenCV Zoo
(Apache 2.0/MIT), NumPy (BSD) e, a partir do M2, MediaPipe (Apache 2.0). O modelo de 68 pontos
do dlib e os modelos do InsightFace **não** devem ser adicionados, pois proíbem uso comercial.
