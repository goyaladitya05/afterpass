import argparse
import sys
import time
from pathlib import Path

from . import bundle, source
from .provenance import Linker


def _shape(text: str) -> tuple[str, list[int]]:
    name, _, dims = text.rpartition("=")
    if not name or not all(d.strip().isdigit() for d in dims.split(",")):
        raise argparse.ArgumentTypeError(f"{text!r}: expected name=d0,d1,... with whole numbers")
    return name, [int(d) for d in dims.split(",")]


def capture(args) -> None:
    t0 = time.perf_counter()
    model = Path(args.model)
    if args.iterations < 1 or args.warmup < 0 or args.decode < 0:
        sys.exit("--iterations must be at least 1, --warmup and --decode at least 0")
    if not model.is_file():
        sys.exit(f"no such file: {model}")
    if model.suffix not in (".onnx", ".xml"):
        sys.exit(f"{model.name}: expected an ONNX model (.onnx) or OpenVINO IR (.xml)")
    src = source.load(model)
    shapes = dict(args.shape)
    if args.runtime == "openvino":
        from . import ov as runtime
    else:
        if args.decode:
            sys.exit("--decode is supported for openvino only")
        if not args.device.startswith("GPU"):
            sys.exit("tensorrt captures need --device GPU")
        try:
            from . import trt as runtime
        except ImportError as e:
            sys.exit(f"TensorRT is not available ({e}): install the extra, for example uv sync --extra tensorrt")
    try:
        extra = {"decode": args.decode} if args.decode else {}
        result = runtime.capture(model, args.device, shapes, args.iterations, args.warmup, Linker(src), **extra)
    except RuntimeError as e:
        sys.exit(f"{args.runtime} failed on {model.name}: {e}")
    meta = {
        "runtime": args.runtime,
        "device": args.device,
        "model": model.name,
        "shapes": shapes,
        "compile_s": result.pop("compile_s"),
        "capture_s": time.perf_counter() - t0,
        # file names only: a bundle is shared, local paths are not
        "command": f"capture {model.name} --runtime {args.runtime} --device {args.device}"
        + "".join(f" --shape {n}={','.join(map(str, d))}" for n, d in args.shape)
        + f" --iterations {args.iterations} --warmup {args.warmup}" + (f" --decode {args.decode}" if args.decode else ""),
    }
    phase = "-decode" if args.decode else ""
    out = Path(args.output or f"{model.stem}.{args.runtime}-{args.device.lower()}{phase}.afterpass.json.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle.write(out, bundle.build(src, result, meta))
    print(f"wrote {out} ({out.stat().st_size} bytes) in {meta['capture_s']:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(prog="afterpass")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="compile a model, profile it and write a bundle")
    c.add_argument("model", help="ONNX model (.onnx) or OpenVINO IR (.xml)")
    c.add_argument("--runtime", choices=["openvino", "tensorrt"], required=True)
    c.add_argument("--device", default="CPU", help="CPU, GPU or an OpenVINO device name such as GPU.1")
    c.add_argument("--shape", type=_shape, action="append", default=[], help="input shape override, name=d0,d1,...")
    c.add_argument("--iterations", type=int, default=50)
    c.add_argument("--warmup", type=int, default=5)
    c.add_argument("--decode", type=int, default=0, metavar="N",
                   help="stateful LLM IR: profile the N single-token steps after a prefill of the --shape input_ids length")
    c.add_argument("-o", "--output", help="bundle path")
    c.set_defaults(func=capture)
    args = p.parse_args()
    args.func(args)
