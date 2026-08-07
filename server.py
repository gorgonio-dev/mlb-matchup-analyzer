"""MLB Matchup Analyzer — cloud web server.

Stdlib-only HTTP server (no Flask needed) exposing the analyzer over the web:

    GET /                → single-page web UI (works on phone + desktop)
    GET /api/teams       → [{id, name}]
    GET /api/analyze     → model probability + breakdown (+ live overlays)
    GET /api/slate       → today's schedule with base model probabilities
    GET /api/odds        → sportsbook odds evaluation (edge, EV, Kelly)
    GET /api/status      → model info, data freshness, updater state
    GET /healthz         → liveness probe (used by keep-warm pings)

A background thread refreshes game data from the MLB Stats API and retrains
the model on boot and every 6 hours — required on free-tier hosts whose disks
reset on every deploy/restart.
"""
import json
import os
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import features  # noqa: F401  (pickled model state resolves against this module)
import odds
import predict

STATUS = {
    "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "model_loaded": False,
    "updating": False,
    "last_update": None,
    "last_update_result": None,
    "last_update_error": None,
}
_update_lock = threading.Lock()


# ---------------------------------------------------------------- background

def _refresh(reason):
    """Pull new games from the MLB Stats API and retrain. Never raises."""
    if not _update_lock.acquire(blocking=False):
        return
    STATUS["updating"] = True
    try:
        import live
        msg = live.update_data(progress=lambda m: None)
        STATUS["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        STATUS["last_update_result"] = f"{msg} ({reason})"
        STATUS["last_update_error"] = None
    except Exception as e:
        STATUS["last_update_error"] = f"{type(e).__name__}: {e}"
    finally:
        STATUS["updating"] = False
        _update_lock.release()


def _background():
    # 1. make sure the model loads (retrains locally on sklearn version skew)
    try:
        predict.load_bundle()
        STATUS["model_loaded"] = True
    except Exception:
        traceback.print_exc()
    # 2. refresh on boot, then every 6 hours
    _refresh("boot")
    STATUS["model_loaded"] = True
    while True:
        time.sleep(6 * 3600)
        _refresh("scheduled")


# ---------------------------------------------------------------- api logic

def api_teams(_q):
    return [{"id": i, "name": n} for i, n in predict.team_names()]


def api_analyze(q):
    t1 = int(q["team1"][0])
    t2 = int(q["team2"][0])
    venue = q.get("venue", ["team1"])[0]
    if t1 == t2:
        return {"error": "Pick two different teams."}
    res = predict.predict_matchup(t1, t2, venue)
    out = {
        "p_team1": round(res["p_team1"], 4),
        "model_name": res["model_name"],
        "as_of": res["as_of"],
        "breakdown": res["breakdown"],
        "sp_context": res.get("sp_context"),
    }
    if q.get("live", ["0"])[0] == "1":
        names = {i: n for i, n in predict.team_names()}
        try:
            import overlays
            lv = overlays.compute_live(t1, t2, names[t1], names[t2],
                                       res["p_team1"], venue=venue)
            out["live"] = {"p_adj": round(lv["p_adj"], 4),
                           "total_shift": round(lv["total_shift"], 4),
                           "lines": lv["lines"]}
        except Exception as e:
            out["live"] = {"error": f"Live data unavailable: {e}"}
    return out


def api_slate(q):
    import live
    names = {i: n for i, n in predict.team_names()}
    games = live.today_schedule()
    out = []
    for g in games:
        h, a = g["home_id"], g["away_id"]
        if h not in names or a not in names:
            continue
        row = {"gamePk": g["gamePk"], "state": g["state"],
               "home_id": h, "away_id": a,
               "home": names[h], "away": names[a],
               "home_pitcher": (g["home_pitcher"] or {}).get("name"),
               "away_pitcher": (g["away_pitcher"] or {}).get("name")}
        try:
            hp = (g["home_pitcher"] or {}).get("id")
            ap = (g["away_pitcher"] or {}).get("id")
            row["p_home"] = round(
                predict.predict_matchup(h, a, "team1", hp, ap)["p_team1"], 4)
        except Exception:
            row["p_home"] = None
        out.append(row)
    return {"date": time.strftime("%Y-%m-%d"), "games": out}


def api_odds(q):
    p = float(q["p"][0])
    ml1 = float(q["ml1"][0])
    ml2 = float(q["ml2"][0]) if q.get("ml2", [""])[0] not in ("", None) else None
    bankroll = float(q["bankroll"][0]) if q.get("bankroll", [""])[0] else None
    return odds.evaluate(p, ml1, ml2, bankroll)


def api_status(_q):
    try:
        bundle = predict.load_bundle()
        model = {"name": bundle["model_name"],
                 "trained_through": bundle["trained_through"],
                 "features": len(bundle["features"])}
    except Exception as e:
        model = {"error": str(e)}
    try:
        import sqlite3
        con = sqlite3.connect(predict.db_path())
        n, latest = con.execute("SELECT COUNT(*), MAX(date) FROM games").fetchone()
        con.close()
        db = {"games": n, "latest_game": latest}
    except Exception as e:
        db = {"error": str(e)}
    try:
        with open(os.path.join(predict.base_dir(), "metrics.json")) as f:
            m = json.load(f)
        holdout = m.get("logistic_2026_holdout") or {}
        metrics = {"holdout_accuracy": holdout.get("accuracy"),
                   "holdout_brier": holdout.get("brier"),
                   "holdout_n": holdout.get("n")}
    except Exception:
        metrics = {}
    return {"model": model, "db": db, "metrics": metrics, **STATUS}


ROUTES = {
    "/api/teams": api_teams,
    "/api/analyze": api_analyze,
    "/api/slate": api_slate,
    "/api/odds": api_odds,
    "/api/status": api_status,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "MLBAnalyzer/1.0"

    def log_message(self, fmt, *args):  # quiet the per-request stderr noise
        pass

    def _send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass  # client hung up before the response finished — not our problem

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        if path == "/healthz":
            return self._send(200, "ok", "text/plain")
        fn = ROUTES.get(path)
        if fn is None:
            return self._send(404, json.dumps({"error": "not found"}),
                              "application/json")
        try:
            q = urllib.parse.parse_qs(parsed.query)
            return self._send(200, json.dumps(fn(q)), "application/json")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return  # client disconnected mid-request; nothing to answer
        except Exception as e:
            traceback.print_exc()
            return self._send(500, json.dumps(
                {"error": f"{type(e).__name__}: {e}"}), "application/json")


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "index.html"), encoding="utf-8") as _f:
    INDEX_HTML = _f.read()


def main():
    port = int(os.environ.get("PORT", "8000"))
    threading.Thread(target=_background, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"MLB Matchup Analyzer listening on :{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
