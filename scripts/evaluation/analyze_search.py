"""Exploratory sensitivity from actual short-budget Optuna trials."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scripts.training.complete_research import write_json


def main():
    reports = ROOT / "reports/completion"
    trials = json.loads((reports / "search_trials.json").read_text())
    if len(trials) != 12:
        raise ValueError("All 12 trials are required for the frozen sensitivity analysis")
    distributions = {"lr0": optuna.distributions.FloatDistribution(.0002, .003, log=True),
                     "weight_decay": optuna.distributions.FloatDistribution(.00001, .002, log=True),
                     "scale": optuna.distributions.FloatDistribution(.05, .4),
                     "imgsz": optuna.distributions.CategoricalDistribution([640, 960])}
    study = optuna.create_study(direction="maximize")
    for t in trials:
        study.add_trial(optuna.trial.create_trial(params=t["params"], distributions=distributions, value=t["value"]))
    importance = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(n_trees=128, max_depth=8, seed=42))
    keys = ["log10_lr0", "log10_weight_decay", "scale", "imgsz"]
    x = np.array([[np.log10(t["params"]["lr0"]), np.log10(t["params"]["weight_decay"]),
                   t["params"]["scale"], t["params"]["imgsz"]] for t in trials])
    y = np.array([t["value"] for t in trials])
    surrogate = RandomForestRegressor(n_estimators=256, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=1)
    cv = cross_val_predict(surrogate, x, y, cv=LeaveOneOut(), n_jobs=1)
    surrogate.fit(x, y)
    grids = [np.linspace(np.log10(.0002), np.log10(.003), 30),
             np.linspace(np.log10(.00001), np.log10(.002), 30), np.linspace(.05, .4, 30), np.array([640, 960])]
    pdp = {}
    for j, key in enumerate(keys):
        predictions = []
        for value in grids[j]:
            modified = x.copy()
            modified[:, j] = value
            predictions.append(float(surrogate.predict(modified).mean()))
        pdp[key] = {"grid": grids[j].tolist(), "partial_dependence": predictions}
    surface = []
    a_grid, b_grid = grids[0][::2], grids[2][::2]
    for a in a_grid:
        values = []
        for b in b_grid:
            modified = x.copy()
            modified[:, 0], modified[:, 2] = a, b
            values.append(float(surrogate.predict(modified).mean()))
        surface.append(values)
    write_json(reports / "search_sensitivity.json", {
        "trials": len(trials), "objective": "validation_mean_AP50_caries_periapical_at_10_epoch_budget",
        "fanova_importance": importance, "surrogate": "RandomForest256_depth4_minleaf2_seed42",
        "leave_one_out": {"r2": float(r2_score(y, cv)), "rmse": float(np.sqrt(mean_squared_error(y, cv))),
                          "observed": y.tolist(), "predicted": cv.tolist()},
        "pdp": pdp, "interaction": {"x": "log10_lr0", "y": "scale", "x_grid": a_grid.tolist(),
                                    "y_grid": b_grid.tolist(), "mean_prediction": surface},
        "interpretation": "Exploratory surrogate on 12 low-budget trials; no causal effects or confirmed optimum. Generalization of the surrogate is assessed by leave-one-out R2; weak or negative R2 makes PDP unreliable."})
    print("Search sensitivity completed")


if __name__ == "__main__":
    main()
