"""Regenerate the benchmark_app fixtures: the toy model's OpenVINO execution graph and statistics report.

usage (from capture/): uv run python ../core/tests/fixtures/make_exec.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

if __name__ == "__main__":
    # relative paths only: the report records the command line, and a fixture holds no local paths
    subprocess.run([sys.executable, "-c", "from openvino.tools.benchmark.main import main; main()",
                    "-m", "toy.onnx", "-d", "CPU", "-hint", "latency", "-niter", "20",
                    "-exec_graph_path", "toy.exec.xml", "-report_type", "no_counters", "-report_folder", "."],
                   cwd=HERE, check=True)
    (HERE / "toy.exec.bin").unlink()  # always empty: an execution graph has no weights
    (HERE / "benchmark_report.csv").rename(HERE / "toy.benchmark_report.csv")
