"""Read evidence to refresh the project memory without loading an ML runtime."""
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path.home() / "AppData/Local/DentalAI"


def optional_json(path, fallback=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else fallback


def test_status(name):
    path=ROOT/"reports"/name
    if not path.is_file():
        return {"available":False}
    suites=list(ET.parse(path).getroot().iter("testsuite"))
    return {"available":True,"passed":sum(int(s.get("tests",0))-int(s.get("skipped",0))-int(s.get("failures",0))-int(s.get("errors",0)) for s in suites),
            "failures":sum(int(s.get("failures",0)) for s in suites),"errors":sum(int(s.get("errors",0)) for s in suites)}


def main():
    runs = []
    for directory in sorted((LOCAL / "runs/research_v2").glob("*")):
        record = directory / "run.json"
        if not record.is_file():
            continue
        item = json.loads(record.read_text(encoding="utf-8"))
        csv = directory / "results.csv"
        epochs = max(0, len(csv.read_text().splitlines()) - 1) if csv.is_file() else 0
        runs.append({"name": directory.name, "status": item["status"], "epochs_recorded": epochs,
                     "elapsed_seconds": item.get("elapsed_seconds"), "best_sha256": item.get("best_sha256"),
                     "resources": item.get("resources", {}), "error": item.get("error")})
    evidence = ROOT / "reports/completion"
    questions_file=ROOT/"docs/defensa/preguntas_oficiales.json"
    official=optional_json(questions_file,{})
    received=isinstance(official,dict) and [q.get("number") for q in official.get("questions",[])]==list(range(1,11))
    confirmations=optional_json(evidence/"confirmation.json",[])
    evaluation=optional_json(evidence/"evaluation.json",{})
    pipeline=optional_json(evidence/"evidence_pipeline.json",{})
    document_qa=optional_json(ROOT/"output/tesis/Verificacion_documental.json",{})
    def document_is_current(stem):
        record=document_qa.get(stem,{})
        if not record.get("source_sha256"):
            return False
        source=ROOT/f"output/tesis/{stem}.md"
        docx,pdf=ROOT/f"output/docx/{stem}.docx",ROOT/f"output/pdf/{stem}.pdf"
        return (all(p.is_file() for p in [source,docx,pdf])
                and hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest()==record["source_sha256"]
                and hashlib.sha256(docx.read_bytes()).hexdigest()==record["sha256_docx"]
                and hashlib.sha256(pdf.read_bytes()).hexdigest()==record["sha256_pdf"])
    stems=["Tesis_Dental_AI_APA7","Guia_de_presentacion_y_entrega_APA7","Guiones_Dental_AI_Presentacion_20261026","Respuestas_Oficiales_Dental_AI_20261026"]
    current_documents=all(document_is_current(stem) for stem in stems)
    visual=optional_json(ROOT/"output/tesis/visual_review_final.json",{})
    def reviewed_file(path, record, hash_key):
        return (path.is_file() and record.get("passed") is True
                and hashlib.sha256(path.read_bytes()).hexdigest()==record.get(hash_key))
    current_visuals=all(reviewed_file(ROOT/f"output/pdf/{stem}.pdf",visual.get(stem,{}),"sha256_pdf")
                        and visual.get(stem,{}).get("reviewed_pages")==list(range(1,document_qa.get(stem,{}).get("pages",0)+1))
                        for stem in stems)
    decks={"presentation":("Dental_AI_Pitch_Semana6_20261026",9),
           "client_presentation":("Dental_AI_Cliente_Semana6_20261026",14)}
    current_decks=all(reviewed_file(ROOT/f"output/presentation/{stem}.pptx",visual.get(key,{}),"sha256_pptx")
                      and visual.get(key,{}).get("reviewed_slides")==list(range(1,count+1))
                      for key,(stem,count) in decks.items())
    state = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "official_questions_received": received,
             "questions_note": "Preguntas proporcionadas por el usuario en docs/defensa/preguntas_oficiales.json." if received else "La profesora aún no las ha enviado; el usuario las compartirá en este chat.",
             "defense_assignment":official.get("readme_order",[]) if received else [],
             "personal_response_recording_exists":official.get("requirements",{}).get("recording_exists",False) if received else False,
             "memory": "CONTEXTO_PROYECTO.md", "runs": runs,
             "measured_progress": {"confirmed_models":len(confirmations),"required_confirmations":6,
                 "evaluated_partitions":list(evaluation.get("partitions",{})),"required_partitions":["train","val","test","external"],
                 "stress_scenarios":len(optional_json(evidence/"stress.json",[])),
                 "models_with_native_parity_passed":sum(bool(r.get("passed")) for r in optional_json(evidence/"metric_parity_all_models.json",{}).values()),
                 "executed_notebooks":optional_json(evidence/"notebooks_executed.json",[]),
                 "quick_tests":test_status("tests_completion_final.xml"),
                 "integration_tests":test_status("tests_completion_integration_final.xml"),
                 "evidence_pipeline_status":pipeline.get("status"),"evidence_pipeline_stage":pipeline.get("current_stage"),
                 "v2_documents_have_source_hash_verification":bool(current_documents),
                 "v2_documents_have_current_visual_review":bool(current_visuals),
                 "presentations_have_current_visual_review":bool(current_decks),
                 "document_pages":{stem:document_qa.get(stem,{}).get("pages") for stem in stems},
                 "browser_final_verified":bool(optional_json(evidence/"browser_final.json",{}).get("actual_final_weights")),
                 "pitch_media_status":optional_json(evidence/"presentation_media.json",{}).get("status")},
             "reports": {name: (evidence / name).is_file() for name in ["selected_configuration.json", "confirmation.json",
                 "evaluation.json", "stress.json", "metric_parity.json", "metric_parity_all_models.json", "export.json", "application_verification.json", "revision_20261026.json"]}}
    target = ROOT / "reports/completion/project_state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"updated_at_utc": state["updated_at_utc"], "completed": sum(r["status"] == "completed" for r in runs),
                      "active": [{"name": r["name"], "epochs": r["epochs_recorded"]} for r in runs if r["status"] == "running"],
                      "reports": state["reports"]}, indent=2))


if __name__ == "__main__":
    main()
