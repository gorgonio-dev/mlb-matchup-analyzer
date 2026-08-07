"""Prediction tracker: logs every prediction, scores it once results arrive,
and reports rolling accuracy, Brier score, calibration, and hypothetical P/L."""
import datetime as dt
import sqlite3

import predict

FLAT_STAKE = 100.0


def _con():
    con = sqlite3.connect(predict.db_path())
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, game_date TEXT,
        team1_id INTEGER, team2_id INTEGER, venue TEXT,
        p_base REAL, p_adj REAL,
        ml_team1 REAL, ml_team2 REAL, bet_side INTEGER, bet_edge REAL,
        outcome INTEGER, scored_gamePk INTEGER)""")
    for col in ("close_ml1 REAL", "close_ml2 REAL"):
        try:
            con.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # column already exists
    return con


def delete_prediction(pred_id):
    con = _con()
    con.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    con.commit(); con.close()


def attach_closing(pred_id, close_ml1, close_ml2):
    """Record the market's closing moneylines for CLV measurement."""
    con = _con()
    con.execute("UPDATE predictions SET close_ml1=?, close_ml2=? WHERE id=?",
                (close_ml1, close_ml2, pred_id))
    con.commit(); con.close()


def _clv(row):
    """Closing-line value for a flagged bet, in probability points.
    Positive = the market moved toward our side after the bet was flagged —
    the strongest early evidence that the model finds real edges."""
    ml1, ml2, bet_side = row[8], row[9], row[10]
    c1, c2 = (row[14], row[15]) if len(row) > 15 else (None, None)
    if not bet_side or None in (ml1, ml2, c1, c2):
        return None
    import odds
    open_fair = odds.novig_two_way(ml1, ml2)
    close_fair = odds.novig_two_way(c1, c2)
    i = 0 if bet_side == 1 else 1
    return close_fair[i] - open_fair[i]


def log_prediction(team1_id, team2_id, venue, p_base, p_adj=None):
    con = _con()
    now = dt.datetime.now()
    cur = con.execute(
        "INSERT INTO predictions(created_at, game_date, team1_id, team2_id, venue,"
        " p_base, p_adj) VALUES (?,?,?,?,?,?,?)",
        (now.isoformat(timespec="seconds"), now.date().isoformat(),
         team1_id, team2_id, venue, p_base, p_adj))
    con.commit(); pid = cur.lastrowid; con.close()
    return pid


def update_adjusted(pred_id, p_adj):
    con = _con()
    con.execute("UPDATE predictions SET p_adj=? WHERE id=?", (p_adj, pred_id))
    con.commit(); con.close()


def attach_odds(pred_id, ml_team1, ml_team2, bet_side, bet_edge):
    """bet_side: 1 or 2 for the side the tool flagged as value, 0 for no bet."""
    con = _con()
    con.execute("UPDATE predictions SET ml_team1=?, ml_team2=?, bet_side=?, bet_edge=? "
                "WHERE id=?", (ml_team1, ml_team2, bet_side, bet_edge, pred_id))
    con.commit(); con.close()


def score_pending():
    """Match unscored predictions to final games (same teams, prediction date or
    the following day) and record outcomes. Returns number newly scored."""
    con = _con()
    rows = con.execute("SELECT id, game_date, team1_id, team2_id FROM predictions "
                       "WHERE outcome IS NULL").fetchall()
    n = 0
    for pid, gdate, t1, t2 in rows:
        nxt = (dt.date.fromisoformat(gdate) + dt.timedelta(days=1)).isoformat()
        g = con.execute(
            "SELECT gamePk, home_id, home_win FROM games WHERE date IN (?,?) AND "
            "((home_id=? AND away_id=?) OR (home_id=? AND away_id=?)) "
            "ORDER BY date LIMIT 1", (gdate, nxt, t1, t2, t2, t1)).fetchone()
        if not g:
            continue
        gamePk, home_id, home_win = g
        team1_won = 1 if ((home_id == t1) == bool(home_win)) else 0
        con.execute("UPDATE predictions SET outcome=?, scored_gamePk=? WHERE id=?",
                    (team1_won, gamePk, pid))
        n += 1
    con.commit(); con.close()
    return n


# predictions table column order:
# 0 id, 1 created_at, 2 game_date, 3 team1_id, 4 team2_id, 5 venue,
# 6 p_base, 7 p_adj, 8 ml_team1, 9 ml_team2, 10 bet_side, 11 bet_edge,
# 12 outcome, 13 scored_gamePk

def _bet_pl(row):
    """Hypothetical P/L for a flat $100 stake on the flagged value side."""
    ml1, ml2, bet_side, outcome = row[8], row[9], row[10], row[12]
    if not bet_side or outcome is None:
        return None
    ml = ml1 if bet_side == 1 else ml2
    if ml is None:
        return None
    won = (outcome == 1) if bet_side == 1 else (outcome == 0)
    dec = 1 + (ml / 100.0 if ml > 0 else 100.0 / -ml)
    return round(FLAT_STAKE * (dec - 1.0), 2) if won else -FLAT_STAKE


def _clv_summary():
    con = _con()
    clvs = [c for c in (_clv(r) for r in con.execute(
        "SELECT * FROM predictions WHERE bet_side IS NOT NULL "
        "AND bet_side > 0").fetchall()) if c is not None]
    con.close()
    if not clvs:
        return {}
    return {"clv_bets": len(clvs), "avg_clv": round(sum(clvs) / len(clvs), 4)}


def summary():
    con = _con()
    rows = con.execute("SELECT * FROM predictions WHERE outcome IS NOT NULL").fetchall()
    pending = con.execute("SELECT COUNT(*) FROM predictions WHERE outcome IS NULL").fetchone()[0]
    con.close()
    if not rows:
        return {"scored": 0, "pending": pending, **_clv_summary()}
    correct = brier = 0.0
    bets = []
    for r in rows:
        p = r[7] if r[7] is not None else r[6]   # p_adj if present else p_base
        p = p if p is not None else 0.5
        y = r[12]
        correct += 1 if (p >= 0.5) == (y == 1) else 0
        brier += (p - y) ** 2
        pl = _bet_pl(r)
        if pl is not None:
            bets.append(pl)
    out = {"scored": len(rows), "pending": pending,
           "accuracy": round(correct / len(rows), 3),
           "brier": round(brier / len(rows), 4)}
    if bets:
        out["bets"] = len(bets)
        out["profit_flat100"] = round(sum(bets), 2)
        out["roi"] = round(sum(bets) / (FLAT_STAKE * len(bets)), 4)
    out.update(_clv_summary())
    return out


def recent(limit=100):
    con = _con()
    rows = con.execute(
        "SELECT p.id, p.game_date, t1.name, t2.name, p.p_base, p.p_adj, p.ml_team1,"
        " p.ml_team2, p.bet_side, p.outcome, p.close_ml1, p.close_ml2 "
        "FROM predictions p JOIN teams t1 ON t1.id=p.team1_id "
        "JOIN teams t2 ON t2.id=p.team2_id ORDER BY p.id DESC LIMIT ?",
        (limit,)).fetchall()
    con.close()
    return rows
