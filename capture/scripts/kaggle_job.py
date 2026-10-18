"""Write a Kaggle script kernel that captures a dense 30B-class LLM on OpenVINO CPU (Kaggle's free CPU notebook).

The kernel prints the machine (CPU, RAM, disk) first, so its limits are measured rather than assumed, then
downloads the IR to scratch disk, captures it with this package (embedded, the repo being private) and writes
the bundle to /kaggle/working.

usage: python scripts/kaggle_job.py <out_dir> <kaggle_username>
       kaggle kernels push -p <out_dir>
       kaggle kernels output <kaggle_username>/afterpass-dense-30b -p <dir>   (when it has finished)
"""

import base64
import io
import json
import sys
import zipfile
from pathlib import Path

REPO = "EmbeddedLLM/Qwen2.5-32B-Instruct-int4-sym-ov"
SHAPES = ["input_ids=1,128", "attention_mask=1,128", "position_ids=1,128", "beam_idx=1"]

JOB = '''import base64, io, resource, shutil, subprocess, sys, zipfile

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True)

sh("lscpu | grep -E 'Model name|^CPU\\\\(s\\\\)|Flags' | cut -c1-300; free -g; df -h / /tmp /kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openvino==2026.4.1", "onnx", "numpy", "huggingface_hub"], check=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(PKG))).extractall("/tmp/afterpass")
if shutil.disk_usage("/tmp").free < 25e9:
    sys.exit("less than 25 GB free on /tmp: the int4 IR does not fit")
from huggingface_hub import snapshot_download
ir = snapshot_download(REPO, local_dir="/tmp/model", allow_patterns=["openvino_model.*", "*.json"])
args = [sys.executable, "-m", "afterpass", "capture", ir + "/openvino_model.xml", "--runtime", "openvino", "--device", "CPU",
        "--iterations", "5", "--warmup", "1", "-o", "/kaggle/working/" + OUT]
for s in SHAPES:
    args += ["--shape", s]
done = subprocess.run(args, env={"PYTHONPATH": "/tmp/afterpass", "PATH": "/usr/bin:/bin"})
print("peak RSS of capture:", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024, "MiB", flush=True)
sh("free -g")
sys.exit(done.returncode)
'''


def main(out: Path, user: str) -> None:
    buf = io.BytesIO()
    src = Path(__file__).resolve().parents[1] / "src"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (src / "afterpass").glob("*.py"):
            z.write(f, f.relative_to(src))
    head = (f"PKG = {base64.b64encode(buf.getvalue()).decode()!r}\nREPO = {REPO!r}\nSHAPES = {SHAPES!r}\n"
            f"OUT = 'qwen2.5-32b.openvino-cpu.afterpass.json.gz'\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "job.py").write_text(head + JOB)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/afterpass-dense-30b", "title": "afterpass dense 30b", "code_file": "job.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
    }, indent=2))
    print(f"wrote {out}/job.py and kernel-metadata.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
