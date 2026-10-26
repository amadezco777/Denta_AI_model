"""Run authorized local evidence stages after all frozen confirmations finish.

No training choices are changed. Failures stop the sequence and remain visible
in the journal. Browser interaction and final visual review remain separate.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/"reports/completion"
RUNTIME = Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def await_confirmations(pid):
    while True:
        source = REPORTS/"confirmation.json"
        confirmations = read(source) if source.is_file() else []
        if len(confirmations) == 6:
            for item in confirmations:
                if read(Path(item["run"])/"run.json")["status"] != "completed":
                    raise ValueError("Confirmation report refers to an incomplete run")
            return
        try:
            command = psutil.Process(pid).cmdline()
        except psutil.NoSuchProcess as exc:
            raise RuntimeError("Experiment process ended before six confirmations; inspect its log") from exc
        if not any("complete_research.py" in part for part in command) or "experiments" not in command:
            raise ValueError("The observed PID is not the expected experiment process")
        print(f"{now()} Waiting for frozen confirmations: {len(confirmations)}/6", flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--from-stage", default="evaluation")
    args = parser.parse_args()
    stages = [
        ("evaluation", [sys.executable, "scripts/evaluation/evaluate_research.py"]),
        ("deployment", [sys.executable, "scripts/evaluation/verify_deployment.py"]),
        ("quick_tests", [str(ROOT/".venv/Scripts/python.exe"), "-m", "pytest", "-q", "--junitxml=reports/tests_completion_final.xml"]),
        ("integration_tests", [sys.executable, "-m", "pytest", "-o", "addopts=", "-m", "integration", "-q", "--junitxml=reports/tests_completion_integration_final.xml"]),
        ("figures", [str(RUNTIME), "output/tesis/completion_figures.py"]),
        ("notebooks", [sys.executable, "scripts/evaluation/build_notebooks.py"]),
        ("guide_and_repository", [str(RUNTIME), "output/tesis/write_project_docs.py"]),
        ("thesis_source", [str(RUNTIME), "output/tesis/rewrite_research.py"]),
    ]
    names = [name for name, _ in stages]
    if args.from_stage not in names:
        raise ValueError(f"Choose --from-stage from {names}")
    journal_path = REPORTS/"evidence_pipeline.json"
    journal = read(journal_path) if journal_path.exists() else {"started_at_utc": now(), "stages": {}}
    journal.update(status="waiting_for_confirmations", observed_training_pid=args.training_pid)
    write(journal_path, journal)
    try:
        await_confirmations(args.training_pid)
        environment = os.environ.copy()
        for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
            environment[key] = "4"
        environment["PYTHONIOENCODING"] = "utf-8"
        for name, command in stages[names.index(args.from_stage):]:
            journal["status"] = "running"
            journal["current_stage"] = name
            record = {"status": "running", "started_at_utc": now(), "command": command}
            journal["stages"][name] = record
            write(journal_path, journal)
            print(f"{now()} START {name}", flush=True)
            result = subprocess.run(command, cwd=ROOT, env=environment)
            record.update(finished_at_utc=now(), returncode=result.returncode,
                          status="completed" if result.returncode == 0 else "failed")
            write(journal_path, journal)
            if result.returncode:
                raise RuntimeError(f"Evidence stage {name} failed; inspect the pipeline log")
            print(f"{now()} COMPLETED {name}", flush=True)
        journal.update(status="completed", finished_at_utc=now(), current_stage=None)
        write(journal_path, journal)
        print("ALL EVIDENCE STAGES COMPLETED; final browser and artifact review remain", flush=True)
    except BaseException as exc:
        journal.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at_utc=now())
        write(journal_path, journal)
        raise


if __name__ == "__main__":
    main()
