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
python -m checkpoint modelos             # baixa os modelos YuNet e SFace (~39 MB, uma vez só)

python -m checkpoint cadastrar 001 "Maria Silva"   # captura 7 amostras do rosto
python -m checkpoint cadastrar 001 --adicionar        # mais amostras (ex.: outra iluminação)
python -m checkpoint listar
python -m checkpoint reconhecer          # identificação ao vivo (Q para sair)
python -m checkpoint remover 001         # apaga o funcionário e seus vetores faciais

python -m checkpoint totem               # modo totem: reconhece, confirma e registra o ponto
python -m checkpoint pin 001             # define o PIN (batida por matrícula + PIN, tecla M)
python -m checkpoint marcacoes           # marcações de hoje (--data AAAA-MM-DD, --todas)
python -m checkpoint verificar           # confere NSR e cadeia de hash das marcações
python -m checkpoint comprovante 1       # salva o PDF do comprovante da marcação NSR 1

python -m checkpoint ponto               # simula a batida sem registrar (teste)
python -m checkpoint vivacidade          # sinais da prova de vida ao vivo, para calibrar

python -m checkpoint benchmark           # tempo de cada etapa, com galeria de 50 pessoas
```

Na batida, depois de reconhecer a pessoa, o sistema sorteia um desafio (piscar ou virar o rosto
para um lado) e, em paralelo, um modelo anti-spoofing avalia se a imagem é de um rosto real ou de
uma foto/tela. A imagem na tela é espelhada, então "esquerda" é o lado esquerdo da tela.
Ajustes em `[liveness]` no `config.toml`.

No totem, cada batida aprovada vira uma marcação com NSR sequencial e hash SHA-256 encadeado à
anterior; a primeira do dia é entrada e as seguintes alternam saída/entrada. O banco recusa
alterar ou apagar marcações, e `verificar` detecta qualquer adulteração feita por fora. Quem não
pode usar a biometria aperta M e digita matrícula e PIN (a marcação fica identificada como
"PIN"); três PINs errados bloqueiam a matrícula por 5 minutos. Batidas repetidas em menos de
2 minutos são ignoradas. Ajustes em `[punch]`.

Depois de cada batida, o totem mostra um QR code por 10 segundos. Lido pelo celular **na mesma
rede**, ele abre o comprovante (página e PDF) servido pelo próprio totem na porta 8080. O link
leva um token secreto, então não dá para abrir o comprovante de outra pessoa trocando o número.
No Windows, na primeira vez, o Firewall pergunta se o Python pode aceitar conexões: libere em
**redes privadas**. Para testar, abra `http://IP-DO-TOTEM:8080/` no celular: deve aparecer
"servidor de comprovantes no ar". Se não abrir, libere a porta no PowerShell como administrador:

```powershell
New-NetFirewallRule -DisplayName "CheckPoint 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private,Public
```

Dados da empresa em `[company]` e ajustes em `[receipt]`.

No cadastro, olhe para a câmera e mexa levemente a cabeça; o sistema recusa se aparecer mais de
uma pessoa, se as amostras forem de pessoas diferentes ou se o rosto já estiver cadastrado.
O limiar de reconhecimento (`match_threshold`) fica no `config.toml`.

O teste da câmera avisa quando a imagem está escura ou o FPS está baixo. Iluminação frontal
no ponto de captura importa mais que a resolução da webcam.

## Testes

```bash
pytest
ruff check .
```

## Licenças de terceiros

Só são usados componentes que permitem uso comercial: OpenCV e modelos YuNet/SFace do OpenCV Zoo
(Apache 2.0/MIT), NumPy (BSD), MediaPipe e o modelo Face Landmarker (Apache 2.0) e os modelos
MiniFASNet do [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
(Apache 2.0), convertidos para ONNX em `src/checkpoint/assets/minifasnet` (ver `NOTICE` ali),
ReportLab e qrcode (BSD). O modelo de 68 pontos
do dlib e os modelos do InsightFace **não** devem ser adicionados, pois proíbem uso comercial.
