"""Estimacion a partir de results.csv de un piloto y la tarifa elegida."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def estimate(results, epochs, hourly_rate, runs=1, setup_minutes=20, storage_cost=0):
    if (type(epochs) is not int or epochs <= 0 or type(runs) is not int or runs <= 0
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                   for v in (hourly_rate, setup_minutes, storage_cost))):
        raise ValueError("Epocas/corridas positivas y costos/tiempos finitos no negativos requeridos")
    with Path(results).open(encoding="utf-8-sig", newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    times = [float(r["time"]) for r in rows]
    if len(times) < 3 or not all(math.isfinite(t) for t in times):
        raise ValueError("Piloto requiere al menos tres epocas y columna time acumulada")
    deltas = [b - a for a, b in zip(times, times[1:])]
    if times[0] < 0 or min(deltas) <= 0:
        raise ValueError("time debe crecer estrictamente")
    seconds_per_epoch = statistics.median(deltas)
    hours = seconds_per_epoch * epochs * runs / 3600 + setup_minutes / 60
    return {"seconds_per_epoch": seconds_per_epoch, "epochs": epochs, "runs": runs,
            "estimated_hours": round(hours, 3), "hourly_rate_usd": hourly_rate,
            "estimated_total_usd": round(hours * hourly_rate + storage_cost, 2),
            "assumption": "Misma GPU, dataset, batch y resolucion que el piloto; sin garantia de tiempo final"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--setup-minutes", type=float, default=20)
    parser.add_argument("--storage-cost", type=float, default=0)
    args = parser.parse_args()
    try:
        print(json.dumps(estimate(args.results, args.epochs, args.hourly_rate, args.runs,
                                  args.setup_minutes, args.storage_cost), indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
