"""Build or verify the explicit source-annotation research view."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.research import build_research_view, verify_research_contract


def main():
    local=Path(os.environ.get("DENTAL_LOCAL_ROOT", Path.home() / "AppData/Local/DentalAI"))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=local / "data/raw_verified")
    parser.add_argument("--out",type=Path,default=local / "data/research_v2")
    parser.add_argument("--contract",type=Path,default=ROOT / "configs/data/contracts/research_v2.json")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    if args.check_only or args.out.exists() or args.contract.exists():
        contract,rows=verify_research_contract(args.contract,for_training=True)
        if Path(contract["root"]).resolve() != args.out.resolve():
            raise ValueError("Contract root does not match the requested output")
        print(json.dumps({"status":"verified_existing_immutable_view","images":len(rows),
                          "reference_status":contract["reference_status"]}))
    else:
        build_research_view(args.source,args.out,args.contract,args.seed)


if __name__=="__main__":
    main()
