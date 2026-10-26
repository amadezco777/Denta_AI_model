"""Create and execute the five evidence notebooks with the real local artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

SETUP = '''from pathlib import Path
import json, sys
ROOT = Path.cwd()
assert (ROOT / "src").is_dir(), "Ejecutar desde la raíz del repositorio"
sys.path.insert(0, str(ROOT))
LOCAL = Path.home() / "AppData/Local/DentalAI"
DATA = LOCAL / "data/research_v2"
REPORTS = ROOT / "reports/completion"
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
from IPython.display import display, Markdown, Image
'''


def build_all():
    notebooks = []
    def notebook(name, title, purpose, cells):
        nb = nbf.v4.new_notebook()
        nb.metadata.kernelspec = {"display_name": "Dental AI investigación", "language": "python", "name": "dental-ai"}
        nb.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nCristhian Pazmino Parra · Universidad Espíritu Santo\n\n{purpose}\n\nReferente: anotaciones publicadas sin adjudicación local. Los datos permanecen fuera del repositorio."), nbf.v4.new_code_cell(SETUP)]
        for kind, text in cells:
            nb.cells.append(nbf.v4.new_code_cell(text) if kind == "code" else nbf.v4.new_markdown_cell(text))
        notebooks.append((name, nb))
    notebook("01_exploracion.ipynb", "Exploración del corpus dental", "Recalcula denominadores sobre la cohorte inmutable y muestra gráficos agregados del EDA ejecutado.", [
        ("code", 'cohort = read(DATA / "cohort_report.json")\nassert cohort["raw_images"] == cohort["accepted_images"] + cohort["excluded_images"]\ndisplay(cohort)'),
        ("code", 'from collections import Counter\nmanifest = [json.loads(s) for s in (DATA / "manifest.jsonl").read_text().splitlines()]\ncounts = Counter(r["split"] for r in manifest)\nassert counts == {k:v["images"] for k,v in cohort["partitions"].items()}\ndisplay(dict(counts))'),
        ("code", 'for name in ["v2_01_cohorte.png", "v2_02_exclusiones.png", "v2_03_clases.png", "v2_04_tamanos.png", "v2_05_coocurrencia.png", "v2_06_resoluciones.png"]:\n    display(Image(filename=str(ROOT / "output/tesis/figures" / name)))'),
        ("markdown", "Las frecuencias son de archivos y cajas, no prevalencias por paciente. La política de exclusión restringe la población evaluada. Fuente: Mureșanu et al. (2025), https://doi.org/10.5281/zenodo.15487430 .")])
    notebook("02_preprocesamiento.ipynb", "Integridad y preprocesamiento", "Verifica los bytes y la separación del manifiesto, sin crear aprobaciones de revisión profesional.", [
        ("code", 'from src.data.research import verify_research_contract, prepare_image, REFERENCE\ncontract, rows = verify_research_contract(ROOT / "configs/data/contracts/research_v2.json", for_training=True)\nassert contract["reference_status"] == REFERENCE\nprint("Contrato de investigación verificado; no acredita aprobación clínica")'),
        ("code", 'from collections import defaultdict\ngroups = defaultdict(set)\nfor row in rows:\n    groups[row["group_id"]].add(row["split"])\nassert all(len(v) == 1 for v in groups.values())\nprint(len(groups), "grupos de similitud; identidad de paciente desconocida")'),
        ("code", 'import cv2, numpy as np\nrow = next(r for r in rows if r["split"] == "val")\nraw = DATA.parent / "raw_verified" / row["source_image"]\nim = cv2.imdecode(np.frombuffer(raw.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)\nderived = cv2.imread(str(DATA / row["image"]), cv2.IMREAD_GRAYSCALE)\nassert np.array_equal(prepare_image(im), derived)\nprint("Derivado idéntico; forma:", derived.shape)'),
        ("markdown", "Reconstrucción desde cero: `src.data.research.build_research_view`. Una nueva política necesita una nueva vista y contrato; el ejecutor rechaza sobrescrituras.")])
    notebook("03_modelado.ipynb", "Entrenamiento y confirmación", "Inspecciona las corridas ejecutadas. Reutiliza sus pesos verificados; no simula haber reentrenado al abrir el cuaderno.", [
        ("code", 'from src.data.contract import sha256_file\nconfirmations = read(REPORTS / "confirmation.json")\nassert len(confirmations) == 6\nsummary = []\nfor item in confirmations:\n    run = Path(item["run"])\n    record = read(run / "run.json")\n    assert record["status"] == "completed"\n    assert sha256_file(run / "weights/best.pt") == record["best_sha256"]\n    summary.append({"run": item["name"], "seed": item["seed"], "validation_mAP50": item["validation"]["map50"], "seconds": record["resources"]["train_seconds"], "peak_gpu_GB": record["resources"]["gpu_mem_peak_gb"]})\ndisplay(summary)'),
        ("code", 'display(Image(filename=str(ROOT / "output/tesis/figures/v2_12_aprendizaje.png")))'),
        ("markdown", "Entrenamiento completo reproducible: `.venv-training\\Scripts\\python.exe scripts/training/complete_research.py experiments`. El protocolo está en `configs/training/research_protocol.yaml`. El runner guarda fallos, configuración efectiva, pesos y recursos. Las comparaciones son de desarrollo; test y externo no se usan para escoger una semilla.")])
    notebook("04_optimizacion.ipynb", "Optimización y sensibilidad", "Analiza los doce ensayos TPE de diez épocas y los diagnósticos del modelo sustituto.", [
        ("code", 'trials = read(REPORTS / "search_trials.json")\nassert len(trials) == 12\nselected = read(REPORTS / "selected_configuration.json")\nassert max(t["value"] for t in trials) == selected["value"]\ndisplay(selected)\ndisplay(trials)'),
        ("code", 'sensitivity = read(REPORTS / "search_sensitivity.json")\ndisplay(sensitivity["fanova_importance"])\ndisplay(sensitivity["leave_one_out"])'),
        ("code", 'for name in ["v2_08_busqueda.png", "v2_09_importancia.png", "v2_10_pdp.png", "v2_11_interaccion.png"]:\n    display(Image(filename=str(ROOT / "output/tesis/figures" / name)))'),
        ("markdown", "Los PDP e interacciones describen un sustituto aprendido con doce ensayos de presupuesto reducido. Un R² bajo o negativo fuera de muestra limita su interpretación. No prueban efectos causales ni un óptimo global. Optuna: Akiba et al. (2019), https://doi.org/10.48550/arXiv.1907.10902 .")])
    notebook("05_evaluacion.ipynb", "Evaluación final del sistema", "Comprueba decisiones congeladas, reservas, incertidumbre, estrés y despliegue con los resultados reales conservados.", [
        ("code", 'evaluation = read(REPORTS / "evaluation.json")\nmetadata = read(ROOT / "models/research_v2/model.json")\nassert metadata["weights_sha256"] == evaluation["decision"]["weights_sha256"]\nassert metadata["threshold"] == evaluation["decision"]["threshold"]\ndisplay({s:{k:v[k] for k in ["images", "boxes", "map50", "map50_95", "false_positives_per_image"]} for s,v in evaluation["partitions"].items()})'),
        ("code", 'display(evaluation["partitions"]["test"]["by_class"])\ndisplay(evaluation["partitions"]["external"]["bootstrap95"])\ndisplay(read(REPORTS / "metric_parity.json"))'),
        ("code", 'stress = read(REPORTS / "stress.json")\nassert len(stress) == 8\ndisplay([{ "scenario": r["scenario"], "delta_map50":r["delta_map50"]} for r in stress])\ndisplay(read(REPORTS / "application_verification.json"))\nassert read(REPORTS / "export.json")["all_parity_checks_passed"]'),
        ("code", 'for name in ["v2_13_ap_holdouts.png", "v2_14_estres.png", "v2_15_calibracion.png", "v2_16_confusion.png"]:\n    display(Image(filename=str(ROOT / "output/tesis/figures" / name)))'),
        ("markdown", "El bootstrap remuestrea imágenes, no pacientes conocidos. Las puntuaciones no son probabilidades de enfermedad. La ausencia de detecciones no descarta patología. Las instrucciones de ejecución están en README y los límites en la ficha de modelo.")])
    return notebooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    kernel_dir = ROOT / "tmp/kernels/dental-ai"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}", "--HistoryManager.enabled=False"], "display_name": "Dental AI", "language": "python"}), encoding="utf-8")
    records = []
    for name, nb in build_all():
        if args.only and name[:2] not in args.only:
            continue
        manager = KernelManager(kernel_name="dental-ai", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
        client = NotebookClient(nb, km=manager, timeout=1200, resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        try:
            client.execute()
        finally:
            nbf.write(nb, ROOT / "notebooks" / name)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
        records.append({"notebook": name, "status": "executed", "code_cells": sum(c.cell_type == "code" for c in nb.cells),
                        "errors": sum(c.cell_type == "code" and any(o.output_type == "error" for o in c.outputs) for c in nb.cells)})
        print(name, "executed", flush=True)
    (ROOT / "reports/completion/notebooks_executed.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
