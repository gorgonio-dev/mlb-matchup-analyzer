"""Train and evaluate the win-probability model.

Time-based validation: train on 2024-2025, test on 2026-to-date.
Final shipped model is retrained on all available games.
Saves model bundle to data/model.pkl and metrics to data/metrics.json.
"""
import json, os, pickle, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

import features as F

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DB = os.path.join(BASE, "games.sqlite")
MODEL = os.path.join(BASE, "model.pkl")
METRICS = os.path.join(BASE, "metrics.json")

MIN_TEAM_GAMES = 15   # drop early-season games (cold features) from training/eval


def to_xy(rows):
    rows = [r for r in rows if r["n_home"] >= MIN_TEAM_GAMES and r["n_away"] >= MIN_TEAM_GAMES]
    X = np.array([[r[f] for f in F.FEATURES] for r in rows])
    y = np.array([r["home_win"] for r in rows])
    return X, y, rows


def evaluate(name, model, X, y, out):
    p = model.predict_proba(X)[:, 1]
    out[name] = {
        "log_loss": round(float(log_loss(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "accuracy": round(float(accuracy_score(y, p > 0.5)), 5),
        "n": int(len(y)),
        "mean_p_home": round(float(p.mean()), 4),
    }
    return p


def main():
    rows, state = F.replay(DB)
    train_rows = [r for r in rows if r["season"] in (2024, 2025)]
    test_rows = [r for r in rows if r["season"] == 2026]
    Xtr, ytr, _ = to_xy(train_rows)
    Xte, yte, te_rows = to_xy(test_rows)

    metrics = {}
    # baselines
    metrics["baseline_home"] = {
        "log_loss": round(float(log_loss(yte, np.full(len(yte), ytr.mean()))), 5),
        "brier": round(float(brier_score_loss(yte, np.full(len(yte), ytr.mean()))), 5),
        "accuracy": round(float((yte == 1).mean()), 5),
        "n": int(len(yte)),
        "note": f"always predict home team at train home rate {ytr.mean():.4f}",
    }

    logit = LogisticRegression(C=1.0, max_iter=2000)
    logit.fit(Xtr, ytr)
    evaluate("logistic_2026_holdout", logit, Xte, yte, metrics)

    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=True, random_state=7)
    gbm.fit(Xtr, ytr)
    evaluate("gbm_2026_holdout", gbm, Xte, yte, metrics)

    gbm_cal = CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=7),
        method="sigmoid", cv=5)
    gbm_cal.fit(Xtr, ytr)
    evaluate("gbm_calibrated_2026_holdout", gbm_cal, Xte, yte, metrics)

    # pick the winner by holdout log loss
    candidates = {
        "logistic": (metrics["logistic_2026_holdout"]["log_loss"], LogisticRegression(C=1.0, max_iter=2000)),
        "gbm_calibrated": (metrics["gbm_calibrated_2026_holdout"]["log_loss"], CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, early_stopping=True, random_state=7),
            method="sigmoid", cv=5)),
    }
    best_name = min(candidates, key=lambda k: candidates[k][0])
    metrics["selected_model"] = best_name

    # calibration table on holdout for the selected architecture
    sel_for_eval = logit if best_name == "logistic" else gbm_cal
    p = sel_for_eval.predict_proba(Xte)[:, 1]
    bins = np.linspace(0.3, 0.75, 10)
    cal = []
    idx = np.digitize(p, bins)
    for b in range(len(bins) + 1):
        m = idx == b
        if m.sum() >= 25:
            cal.append({"pred": round(float(p[m].mean()), 3),
                        "actual": round(float(yte[m].mean()), 3), "n": int(m.sum())})
    metrics["calibration_2026"] = cal

    # final model: retrain selected architecture on ALL data
    Xall, yall, _ = to_xy(rows)
    final = candidates[best_name][1]
    final.fit(Xall, yall)

    if best_name == "logistic":
        metrics["coefficients"] = dict(zip(F.FEATURES, [round(float(c), 5) for c in final.coef_[0]]))
        metrics["intercept"] = round(float(final.intercept_[0]), 5)

    with open(MODEL, "wb") as f:
        pickle.dump({"model": final, "features": F.FEATURES, "state": state,
                     "trained_through": state["as_of"], "model_name": best_name}, f)
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
