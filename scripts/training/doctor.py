"""Diagnostico local; no descarga pesos ni modifica la configuracion de CUDA."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def diagnose(check_torch=False):
    report = {"python": sys.version, "packages": {}, "gpu": [], "data_images": {}, "torch": None}
    for package in ("torch", "torchvision", "ultralytics", "label-studio"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True)
        report["gpu"] = gpu.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        report["gpu_error"] = str(exc)
    for folder in ("raw", "interim", "processed"):
        report["data_images"][folder] = sum(1 for p in (ROOT / "data" / folder).rglob("*")
                                                if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if check_torch:
        code = ("import json,torch; d={'version':torch.__version__,'cuda_build':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available()}; "
                "d.update({'arch_list':torch.cuda.get_arch_list(),'device':torch.cuda.get_device_name(0),"
                "'capability':torch.cuda.get_device_capability(0),'tensor_test':float(torch.ones(4,device='cuda').sum())} "
                "if torch.cuda.is_available() else {}); print(json.dumps(d))")
        try:
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
            report["torch"] = json.loads(result.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            report["torch"] = {"error": str(exc), "details": (getattr(exc, "stderr", "") or "")[-1500:]}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = json.dumps(diagnose(args.torch), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    print(output)
