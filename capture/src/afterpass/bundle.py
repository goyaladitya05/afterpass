"""Bundle format v0: gzip-compressed JSON. Spec in notes/bundle-format.md."""

import datetime
import gzip
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import onnx

from . import __version__

SCHEMA_VERSION = 0


def _cpu() -> dict:
    info = {"name": platform.processor(), "flags": [], "cores": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "model name":
                info["name"] = value.strip()
            elif key == "flags":
                info["flags"] = value.split()
                break
    except OSError:
        pass
    return info


def _os() -> str:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    return platform.platform()


def _gpus() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [dict(zip(("name", "driver", "memory"), (f.strip() for f in line.split(",")))) for line in out.splitlines() if line]


def environment() -> dict:
    return {
        "os": _os(),
        "kernel": platform.release(),
        "cpu": _cpu(),
        "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "gpus": _gpus(),
        "python": platform.python_version(),
        "onnx": onnx.__version__,
        "afterpass": __version__,
    }


def build(source: dict, result: dict, capture: dict) -> dict:
    return {
        "format": "afterpass-bundle",
        "schema_version": SCHEMA_VERSION,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "capture": capture,
        "environment": environment(),
        "source": source,
        "compiled": result["compiled"],
        "profile": result["profile"],
        "provenance": result["links"],
    }


def write(path: Path, bundle: dict) -> None:
    data = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False).encode()
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(data)


def read(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        bundle = json.loads(f.read())
    if bundle.get("format") != "afterpass-bundle":
        raise ValueError(f"{path} is not an afterpass bundle")
    return bundle
