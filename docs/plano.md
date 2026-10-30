# Plano do protótipo

## Escopo

| Decisão | Valor |
| --- | --- |
| Finalidade | Protótipo / controle interno (não é REP-P oficial) |
| Pontos de coleta | 1 totem |
| Funcionários | até 50 |
| Banco | SQLite (módulo `sqlite3` do Python; migração a PostgreSQL só se houver mais totens) |
| Detecção facial | OpenCV YuNet |
| Reconhecimento | OpenCV SFace (embedding de 128 posições, similaridade de cosseno) |
| Prova de vida | Desafio aleatório (MediaPipe Face Landmarker) + anti-spoofing de textura |
| Recibo | PDF simples + QR code (sem assinatura ICP-Brasil no protótipo) |

Mesmo no protótipo, mantemos as decisões que deixam o caminho aberto para o REP-P: marcações
append-only com NSR e hash SHA-256 encadeado, e relógio com hora em segundos na tela.

## Arquitetura

```
webcam -> captura (OpenCV) -> detecção (YuNet) -> prova de vida (Face Landmarker + anti-spoofing)
       -> embedding (SFace) -> busca 1:N em memória -> registro (SQLite) -> comprovante (PDF + QR)
```

O cadastro guarda só dados do funcionário e o vetor facial cifrado (AES-256-GCM). Fotos não são
armazenadas.

## Milestones

| # | Entrega | Critério de aceite |
| --- | --- | --- |
| M0 | Estrutura, configuração, CLI e diagnóstico da webcam | `python -m checkpoint camera` mede resolução, FPS e brilho da C270 |
| M1 | Detecção YuNet, embedding SFace, cadastro (7 capturas) e identificação 1:N | Benchmark por etapa no i3; ≥ 95% de acerto com 5 pessoas e nenhum falso aceite em 100 tentativas de não cadastrados |
| M2 | Prova de vida com desafio aleatório e anti-spoofing | Foto impressa e no celular barradas em 100%; vídeo no celular em ≥ 90%; pessoa real aprovada em até 3 s |
| M3 | Marcações append-only com NSR e hash encadeado, tela do totem, alternativa por matrícula + PIN | Batida em até 2 s após a prova de vida; adulteração no banco detectada |
| M4 | Comprovante em PDF com QR, servido na rede local | QR lido no celular abre o comprovante |
| M5 | Painel web do RH (cadastro, espelho de ponto, exportação CSV) | RH cadastra, consulta e exporta sem usar o terminal |
| M6 | Criptografia dos vetores, logs de auditoria, backup, aviso de privacidade (LGPD) e pacote para Ubuntu | Instalação limpa no Ubuntu em menos de 30 min; banco copiado não revela os vetores |

Fora do escopo do protótipo (para quando virar ponto oficial): AFD e AEJ nos leiautes da
Portaria 671, assinatura ICP-Brasil, sincronização com a Hora Legal Brasileira e registro no INPI.

## LGPD

Biometria facial é dado pessoal sensível (art. 5º, II). Mesmo em uso interno: base legal no
art. 11, II, "g", aviso de privacidade no cadastro, alternativa sem biometria, exclusão do vetor
no desligamento e nenhum armazenamento de fotos.
