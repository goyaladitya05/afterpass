"""Export the M0 corpus models to ONNX with static shapes.

usage: uv run --group corpus python scripts/export_corpus.py <corpus_dir> [resnet50 bert yolo llm]
"""

import shutil
import sys
from pathlib import Path

import torch

OPSET = 17
SEQ = 128


def export(model, args, path: Path, input_names, output_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, args, str(path), input_names=input_names, output_names=output_names,
                          opset_version=OPSET, dynamo=False)
    print(f"wrote {path}")


def resnet50(out: Path):
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval()
    export(m, (torch.rand(1, 3, 224, 224),), out / "resnet50" / "resnet50.onnx", ["input"], ["logits"])


def bert(out: Path):
    from transformers import BertModel
    m = BertModel.from_pretrained("google-bert/bert-base-uncased", attn_implementation="eager").eval()
    ids = torch.ones(1, SEQ, dtype=torch.int64)
    export(m, (ids, torch.ones_like(ids), torch.zeros_like(ids)), out / "bert-base" / "bert-base.onnx",
           ["input_ids", "attention_mask", "token_type_ids"], ["last_hidden_state", "pooler_output"])


def yolo(out: Path):
    from ultralytics import YOLO
    dst = out / "yolo11n"
    dst.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(dst / "yolo11n.pt"))
    shutil.move(m.export(format="onnx", imgsz=640, opset=OPSET, dynamic=False, simplify=True), dst / "yolo11n.onnx")
    print(f"wrote {dst / 'yolo11n.onnx'}")


def llm(out: Path):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16,
                                             attn_implementation="eager", use_cache=False).eval()
    ids = torch.ones(1, SEQ, dtype=torch.int64)
    export(m, (ids, torch.ones_like(ids)), out / "tinyllama-1.1b" / "tinyllama-1.1b.onnx",
           ["input_ids", "attention_mask"], ["logits"])


if __name__ == "__main__":
    corpus = Path(sys.argv[1])
    for name in sys.argv[2:] or ["resnet50", "bert", "yolo", "llm"]:
        globals()[name](corpus)
