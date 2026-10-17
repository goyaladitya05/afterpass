import argparse
import sys
import time
from pathlib import Path

from . import bundle, source
from .provenance import Linker


def _shape(text: str) -> tuple[str, list[int]]:
    name, _, dims = text.rpartition("=")
    if not name:
        raise argparse.ArgumentTypeError("expected name=d0,d1,...")
    return name, [int(d) for d in dims.split(",")]


def capture(args) -> None:
    t0 = time.perf_counter()
    model = Path(args.model)
    src = source.load(model)
    shapes = dict(args.shape)
    if args.runtime == "openvino":
        from . import ov as runtime
    else:
        from . import trt as runtime
        if not args.device.startswith("GPU"):
            sys.exit("tensorrt captures need --device GPU")
    result = runtime.capture(model, args.device, shapes, args.iterations, args.warmup, Linker(src))
    meta = {
        "runtime": args.runtime,
        "device": args.device,
        "model": model.name,
        "shapes": shapes,
        "compile_s": result.pop("compile_s"),
        "capture_s": time.perf_counter() - t0,
        "command": " ".join(sys.argv[1:]),
    }
    out = Path(args.output or f"{model.stem}.{args.runtime}-{args.device.lower()}.afterpass.json.gz")
    bundle.write(out, bundle.build(src, result, meta))
    print(f"wrote {out} ({out.stat().st_size} bytes) in {meta['capture_s']:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(prog="afterpass")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="compile a model, profile it and write a bundle")
    c.add_argument("model", help="ONNX model")
    c.add_argument("--runtime", choices=["openvino", "tensorrt"], required=True)
    c.add_argument("--device", default="CPU", help="CPU, GPU or an OpenVINO device name such as GPU.1")
    c.add_argument("--shape", type=_shape, action="append", default=[], help="input shape override, name=d0,d1,...")
    c.add_argument("--iterations", type=int, default=50)
    c.add_argument("--warmup", type=int, default=5)
    c.add_argument("-o", "--output", help="bundle path")
    c.set_defaults(func=capture)
    args = p.parse_args()
    args.func(args)
