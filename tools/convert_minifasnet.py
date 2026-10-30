"""Converte os pesos PyTorch do MiniFASNet para ONNX.

Uso único, só para quem quiser reproduzir os arquivos em src/checkpoint/assets/minifasnet.
Requer torch e onnx (não são dependências do projeto):

    git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    pip install torch onnx
    python tools/convert_minifasnet.py Silent-Face-Anti-Spoofing src/checkpoint/assets/minifasnet
"""

import collections
import sys
from pathlib import Path

import torch

MODELS = [
    ("MiniFASNetV2", "2.7_80x80_MiniFASNetV2.pth", "minifasnet_v2_2.7_80x80.onnx"),
    ("MiniFASNetV1SE", "4_0_0_80x80_MiniFASNetV1SE.pth", "minifasnet_v1se_4.0_80x80.onnx"),
]


def main(repo: Path, out_dir: Path) -> None:
    sys.path.insert(0, str(repo / "src" / "model_lib"))
    import MiniFASNet

    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name, weights, onnx_name in MODELS:
        model = getattr(MiniFASNet, class_name)(conv6_kernel=(5, 5))
        state = torch.load(repo / "resources" / "anti_spoof_models" / weights, map_location="cpu")
        state = collections.OrderedDict(
            (k.removeprefix("module."), v) for k, v in state.items()
        )
        model.load_state_dict(state)
        model.eval()
        torch.onnx.export(model, torch.zeros(1, 3, 80, 80), out_dir / onnx_name,
                          input_names=["input"], output_names=["logits"], opset_version=13,
                          dynamo=False)
        print(f"{onnx_name} gerado")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
