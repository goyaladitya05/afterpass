"""Bundle format v0: gzip-compressed JSON."""

import datetime
import gzip
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import onnx

from . import __version__

SCHEMA_VERSION = 0


def _cpu() -> dict:
    info = {"name": platform.processor(), "flags": [], "cores": os.cpu_count()}
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            info["name"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        return info  # no ISA flag list on Windows without extra dependencies
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


def _ram() -> int:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(f, ctypes.c_ulonglong) for f in (
                "total", "avail", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_ext")]

        m = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.total
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _os() -> str:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    return platform.platform()


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout if r.returncode == 0 else ""


def _gpus() -> list[dict]:
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json"])
        rows = json.loads(out) if out.strip() else []
        return [{"name": r["Name"], "driver": r["DriverVersion"]} for r in (rows if isinstance(rows, list) else [rows])]
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return [dict(zip(("name", "driver", "memory"), (f.strip() for f in line.split(",")))) for line in out.splitlines() if line]


def environment() -> dict:
    return {
        "os": _os(),
        "kernel": platform.version() if sys.platform == "win32" else platform.release(),
        "cpu": _cpu(),
        "ram_bytes": _ram(),
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
