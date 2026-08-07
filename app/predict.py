"""Prediction layer: load model bundle, produce matchup probabilities + breakdown."""
import os, pickle, sqlite3, sys
import numpy as np

import features as F


def base_dir():
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        # macOS .app bundle: executable lives in Foo.app/Contents/MacOS;
        # the data folder sits NEXT TO the .app, not inside it
        norm = exe_dir.replace("\\", "/")
        if norm.endswith(".app/Contents/MacOS"):
            exe_dir = os.path.dirname(os.path.dirname(os.path.dirname(exe_dir)))
        return os.path.join(exe_dir, "data")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def db_path():
    return os.path.join(base_dir(), "games.sqlite")


def model_path():
    return os.path.join(base_dir(), "model.pkl")


_bundle = None


def load_bundle(force=False):
    global _bundle
    if _bundle is None or force:
        try:
            with open(model_path(), "rb") as f:
                _bundle = pickle.load(f)
        except Exception:
            # shipped pickle incompatible with this machine's library versions
            # (e.g. much older scikit-learn): retrain natively from the bundled
            # database — one-time, a minute or two — then load the fresh model
            import train
            train.main()
            with open(model_path(), "rb") as f:
                _bundle = pickle.load(f)
    return _bundle


def team_names():
    con = sqlite3.connect(db_path())
    rows = con.execute("SELECT id, name FROM teams ORDER BY name").fetchall()
    con.close()
    return rows


def _prob_home(bundle, home_id, away_id, home_pid=None, away_pid=None):
    feats, extra = F.matchup_features(bundle["state"], home_id, away_id,
                                      home_pid=home_pid, away_pid=away_pid)
    X = np.array([[feats.get(f, 0.0) for f in bundle["features"]]])
    return float(bundle["model"].predict_proba(X)[0, 1]), extra


def model_knows_starters():
    """True when the loaded model was trained with starter features."""
    return "starter_ra_diff" in load_bundle()["features"]


def predict_matchup(team1_id, team2_id, venue="team1",
                    team1_pid=None, team2_pid=None):
    """venue: 'team1' (game at team1's park), 'team2', or 'neutral'.
    Optional starter IDs feed the model's native pitcher feature.
    Returns dict with p_team1 (prob team1 wins) and a stats breakdown."""
    bundle = load_bundle()
    if venue == "team1":
        p_home, extra = _prob_home(bundle, team1_id, team2_id, team1_pid, team2_pid)
        p1 = p_home
    elif venue == "team2":
        p_home, extra = _prob_home(bundle, team2_id, team1_id, team2_pid, team1_pid)
        p1 = 1 - p_home
    else:  # neutral: average both orientations
        pa, extra = _prob_home(bundle, team1_id, team2_id, team1_pid, team2_pid)
        pb, _ = _prob_home(bundle, team2_id, team1_id, team2_pid, team1_pid)
        p1 = (pa + (1 - pb)) / 2

    return {
        "p_team1": p1,
        "sp_context": extra,
        "as_of": bundle["trained_through"],
        "model_name": bundle["model_name"],
        "breakdown": {
            "team1": team_summary(bundle, team1_id),
            "team2": team_summary(bundle, team2_id),
            "h2h": h2h_summary(bundle, team1_id, team2_id),
        },
    }


def team_summary(bundle, tid):
    t = bundle["state"]["teams"][tid]
    return {
        "record": f"{t.w}-{t.l}",
        "win_pct": round(t.w / t.n, 3) if t.n else 0.5,
        "run_diff_pg": round((t.rf - t.ra) / t.n, 2) if t.n else 0.0,
        "runs_scored_pg": round(t.rf / t.n, 2) if t.n else 0.0,
        "runs_allowed_pg": round(t.ra / t.n, 2) if t.n else 0.0,
        "last10": f"{sum(t.last10)}-{len(t.last10) - sum(t.last10)}",
        "home_record": f"{t.home_w}-{t.home_l}",
        "away_record": f"{t.away_w}-{t.away_l}",
        "elo": round(t.elo, 1),
        "prior_season_wpct": round(t.prior_wpct, 3),
    }


def h2h_summary(bundle, team1_id, team2_id):
    key = (min(team1_id, team2_id), max(team1_id, team2_id))
    dq = bundle["state"]["h2h"].get(key, [])
    o = F._date_ord(bundle["trained_through"]) + 1
    dq = [x for x in dq if o - x[0] <= F.H2H_WINDOW_DAYS]
    w1 = sum(1 for _, wnr in dq if wnr == team1_id)
    return {"meetings_2yr": len(dq), "team1_wins": w1, "team2_wins": len(dq) - w1}


def latest_game_date():
    con = sqlite3.connect(db_path())
    d = con.execute("SELECT MAX(date) FROM games").fetchone()[0]
    con.close()
    return d
