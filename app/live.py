"""
Live data layer — used on the user's machine with normal internet access.

- update_data(): pulls all new final game results from the MLB Stats API,
  upserts them into the local database, and retrains the model.
- probable_pitchers(): today's scheduled starters for a matchup.
- pitcher_line(): a starter's season pitching stats (shrunk ERA for the overlay).
- pitcher_vs_team(): the starter's career line against a specific opponent.
- apply_pitcher_overlay(): documented adjustment of the base win probability
  for the announced starting-pitcher gap.

All functions raise LiveDataError on network failure; callers degrade gracefully.
"""
import datetime as dt
import sqlite3

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

import predict

API = "https://statsapi.mlb.com/api/v1"
LEAGUE_AVG_ERA = 4.10
ERA_SHRINK_IP = 40.0      # innings of league-average shrinkage
OVERLAY_LOGIT_PER_RUN = 0.18   # logit shift per run of shrunk-ERA gap
OVERLAY_CAP = 0.12             # max probability shift from the overlay


class LiveDataError(Exception):
    pass


# small in-memory cache so repeated Analyze clicks don't refetch rosters
_cache = {}
CACHE_TTL = 600.0


def _cached(kind, key, fn):
    import time
    now = time.time()
    hit = _cache.get((kind, key))
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    val = fn()
    _cache[(kind, key)] = (now, val)
    return val


def _get(path, params):
    if requests is None:
        raise LiveDataError("The 'requests' package is not installed.")
    try:
        r = requests.get(f"{API}/{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise LiveDataError(f"MLB Stats API unreachable: {e}") from e


# ---------------- data update ----------------

def update_data(progress=lambda msg: None):
    """Fetch all final games newer than the local DB and retrain. Returns summary str."""
    con = sqlite3.connect(predict.db_path())
    last = con.execute("SELECT MAX(date) FROM games").fetchone()[0]
    start = dt.date.fromisoformat(last) - dt.timedelta(days=3)  # re-scan recent days
    today = dt.date.today()
    n_new = 0
    cur = start
    while cur <= today:
        end = min(cur + dt.timedelta(days=29), today)
        progress(f"Fetching {cur} .. {end}")
        data = _get("schedule", {
            "sportId": 1, "startDate": cur.isoformat(), "endDate": end.isoformat(),
            "gameType": "R", "hydrate": "probablePitcher",
            "fields": "dates,date,games,gamePk,officialDate,status,codedGameState,"
                      "teams,away,home,team,id,score,isWinner,probablePitcher"})
        con.execute("CREATE TABLE IF NOT EXISTS starters("
                    "gamePk INTEGER PRIMARY KEY, home_pid INTEGER, away_pid INTEGER)")
        for day in data.get("dates", []):
            for g in day["games"]:
                t = g.get("teams", {})
                a, h = t.get("away", {}), t.get("home", {})
                hp = (h.get("probablePitcher") or {}).get("id")
                ap = (a.get("probablePitcher") or {}).get("id")
                if hp or ap:
                    con.execute("INSERT OR REPLACE INTO starters VALUES (?,?,?)",
                                (g["gamePk"], hp, ap))
                if g.get("status", {}).get("codedGameState") != "F":
                    continue
                if "score" not in a or "score" not in h:
                    continue
                if (h["score"] > a["score"]) != bool(h.get("isWinner")):
                    continue  # inconsistent row; skip
                con.execute(
                    "INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?)",
                    (g["gamePk"], g["officialDate"], int(g["officialDate"][:4]),
                     a["team"]["id"], h["team"]["id"], a["score"], h["score"],
                     1 if h.get("isWinner") else 0))
                n_new += 1
        con.commit()
        cur = end + dt.timedelta(days=1)
    con.close()

    progress("Retraining model on updated data...")
    import train
    train.main()
    predict.load_bundle(force=True)
    return f"Update complete: {n_new} game rows refreshed; model retrained."


# ---------------- pitchers ----------------

def probable_pitchers(team1_id, team2_id, date=None):
    """If these teams play on `date` (default today), return probable starters:
    {'home_id':..,'away_id':..,'home': {'id','name'}|None, 'away': {...}|None}"""
    d = (date or dt.date.today()).isoformat()
    data = _get("schedule", {
        "sportId": 1, "startDate": d, "endDate": d, "hydrate": "probablePitcher",
        "fields": "dates,games,gamePk,teams,away,home,team,id,probablePitcher,fullName"})
    for day in data.get("dates", []):
        for g in day.get("games", []):
            t = g["teams"]
            ids = {t["home"]["team"]["id"], t["away"]["team"]["id"]}
            if ids == {team1_id, team2_id}:
                def pp(side):
                    p = t[side].get("probablePitcher")
                    return {"id": p["id"], "name": p.get("fullName", "?")} if p else None
                return {"home_id": t["home"]["team"]["id"],
                        "away_id": t["away"]["team"]["id"],
                        "home": pp("home"), "away": pp("away")}
    return None


def pitcher_line(pid, season=None):
    season = season or dt.date.today().year
    data = _get(f"people/{pid}/stats", {
        "stats": "season", "group": "pitching", "season": season,
        "fields": "stats,splits,stat,era,whip,inningsPitched,strikeOuts,"
                  "baseOnBalls,wins,losses,gamesStarted"})
    try:
        s = data["stats"][0]["splits"][0]["stat"]
    except (KeyError, IndexError):
        return None
    ip = _ip_to_float(s.get("inningsPitched", "0"))
    era = float(s.get("era", LEAGUE_AVG_ERA))
    shrunk = (era * ip + LEAGUE_AVG_ERA * ERA_SHRINK_IP) / (ip + ERA_SHRINK_IP)
    return {"era": era, "era_shrunk": round(shrunk, 2), "whip": s.get("whip"),
            "ip": s.get("inningsPitched"), "so": s.get("strikeOuts"),
            "bb": s.get("baseOnBalls"), "record": f"{s.get('wins', 0)}-{s.get('losses', 0)}",
            "gs": s.get("gamesStarted")}


def pitcher_vs_team(pid, opposing_team_id):
    """Career batting-against line for this pitcher vs a specific team
    (how that opponent has historically hit him)."""
    data = _get(f"people/{pid}/stats", {
        "stats": "vsTeamTotal", "group": "pitching", "opposingTeamId": opposing_team_id,
        "fields": "stats,splits,stat,avg,ops,strikeOuts,homeRuns,atBats,hits"})
    try:
        s = data["stats"][0]["splits"][0]["stat"]
    except (KeyError, IndexError):
        return None
    if not s.get("atBats"):
        return None
    return {"avg_against": s.get("avg"), "ops_against": s.get("ops"),
            "so": s.get("strikeOuts"), "hr": s.get("homeRuns"), "ab": s.get("atBats")}


def _ip_to_float(ip_str):
    # "123.2" means 123 and 2/3 innings
    try:
        whole, _, frac = str(ip_str).partition(".")
        return int(whole) + (int(frac) / 3 if frac else 0)
    except ValueError:
        return 0.0


# ---------------- today's schedule (slate board) ----------------

def today_schedule(date=None):
    """All MLB games scheduled for `date` (default today) with probable pitchers."""
    d = (date or dt.date.today()).isoformat()
    data = _get("schedule", {
        "sportId": 1, "startDate": d, "endDate": d, "hydrate": "probablePitcher",
        "fields": "dates,games,gamePk,status,codedGameState,teams,away,home,"
                  "team,id,probablePitcher,fullName"})
    games = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            t = g["teams"]
            def pp(side):
                p = t[side].get("probablePitcher")
                return {"id": p["id"], "name": p.get("fullName", "?")} if p else None
            games.append({"gamePk": g["gamePk"],
                          "state": g.get("status", {}).get("codedGameState", ""),
                          "home_id": t["home"]["team"]["id"],
                          "away_id": t["away"]["team"]["id"],
                          "home_pitcher": pp("home"), "away_pitcher": pp("away")})
    return games


# ---------------- bullpen quality ----------------

LEAGUE_AVG_PEN_ERA = 4.00
PEN_SHRINK_IP = 30.0          # innings of league-average shrinkage per reliever
PEN_LOGIT_PER_RUN = 0.10      # bullpen covers ~1/3 of innings
PEN_CAP = 0.05                # max probability shift from the bullpen overlay


def players_pitching(pids, season=None):
    """Batched season pitching stats. Returns {pid: {era, ip, gs, g}}."""
    season = season or dt.date.today().year
    out = {}
    pids = list(pids)
    for i in range(0, len(pids), 40):
        chunk = pids[i:i + 40]
        data = _get("people", {
            "personIds": ",".join(map(str, chunk)),
            "hydrate": f"stats(group=[pitching],type=[season],season={season})",
            "fields": "people,id,stats,splits,team,stat,era,inningsPitched,"
                      "gamesPlayed,gamesStarted"})
        for p in data.get("people", []):
            splits = []
            for st in p.get("stats", []):
                splits.extend(st.get("splits", []))
            if not splits:
                continue
            combined = [s for s in splits if "team" not in s]
            use = combined if combined else splits
            ip = sum(_ip_to_float(s["stat"].get("inningsPitched", "0")) for s in use)
            if ip <= 0:
                continue
            era_num = sum(float(s["stat"].get("era", 0) or 0) *
                          _ip_to_float(s["stat"].get("inningsPitched", "0")) for s in use)
            out[p["id"]] = {"era": era_num / ip, "ip": ip,
                            "gs": sum(int(s["stat"].get("gamesStarted", 0) or 0) for s in use),
                            "g": sum(int(s["stat"].get("gamesPlayed", 0) or 0) for s in use)}
    return out


def bullpen_strength(team_id, season=None):
    """IP-weighted, shrunk ERA of the relievers on the current active roster."""
    def build():
        roster = team_roster(team_id)
        arms = [r for r in roster if r["pos"] == "P"]
        stats = players_pitching([a["id"] for a in arms], season)
        ip_sum = era_sum = 0.0
        n = 0
        for a in arms:
            s = stats.get(a["id"])
            if not s or s["ip"] < 5:
                continue
            if s["g"] > 0 and s["gs"] / s["g"] >= 0.5:
                continue  # starter, not bullpen
            shrunk = (s["era"] * s["ip"] + LEAGUE_AVG_PEN_ERA * PEN_SHRINK_IP) / (s["ip"] + PEN_SHRINK_IP)
            era_sum += shrunk * s["ip"]
            ip_sum += s["ip"]
            n += 1
        if ip_sum <= 0:
            return None
        return {"era_shrunk": round(era_sum / ip_sum, 2), "relievers": n,
                "ip": round(ip_sum, 1)}
    return _cached("bullpen", (team_id, season), build)


def apply_bullpen_overlay(p_home, home_pen, away_pen):
    """Shift home win probability for the bullpen-quality gap. Capped +/-5 pts."""
    import math
    if not home_pen or not away_pen:
        return p_home, 0.0
    gap = away_pen["era_shrunk"] - home_pen["era_shrunk"]  # positive favors home
    logit = math.log(p_home / (1 - p_home)) + PEN_LOGIT_PER_RUN * gap
    p_new = 1 / (1 + math.exp(-logit))
    shift = max(-PEN_CAP, min(PEN_CAP, p_new - p_home))
    return p_home + shift, shift


# ---------------- roster-aware lineup strength ----------------

LEAGUE_AVG_OPS = 0.700
OPS_SHRINK_PA = 100.0        # plate appearances of league-average shrinkage
OPS_TO_RUNS = 13.0           # runs/game per 1.000 of team OPS (empirical slope)
ROSTER_LOGIT_PER_RUN = 0.25  # logit shift per run/game of lineup delta
ROSTER_CAP = 0.08            # max probability shift from the roster overlay
MIN_PA = 20                  # ignore hitters with almost no season sample


def team_roster(team_id):
    def fetch():
        data = _get(f"teams/{team_id}/roster", {
            "rosterType": "active",
            "fields": "roster,person,id,fullName,position,abbreviation"})
        return [{"id": r["person"]["id"], "name": r["person"]["fullName"],
                 "pos": r.get("position", {}).get("abbreviation", "")}
                for r in data.get("roster", [])]
    return _cached("roster", team_id, fetch)


def players_hitting(pids, season=None):
    """Batched season hitting stats, carried with the player across teams.
    Returns {pid: {ops, pa, teams: [team names PA was earned with]}}."""
    season = season or dt.date.today().year
    out = {}
    pids = list(pids)
    for i in range(0, len(pids), 40):
        chunk = pids[i:i + 40]
        data = _get("people", {
            "personIds": ",".join(map(str, chunk)),
            "hydrate": f"stats(group=[hitting],type=[season],season={season})",
            "fields": "people,id,fullName,stats,splits,team,name,stat,"
                      "plateAppearances,ops"})
        for p in data.get("people", []):
            splits = []
            for st in p.get("stats", []):
                splits.extend(st.get("splits", []))
            if not splits:
                continue
            # a traded player has one split per team plus a combined split
            # (the combined one carries no "team" attribute) — prefer it
            combined = [s for s in splits if "team" not in s]
            use = combined if combined else splits
            pa = sum(int(s["stat"].get("plateAppearances", 0) or 0) for s in use)
            if pa <= 0:
                continue
            ops = sum(float(s["stat"].get("ops", 0) or 0) *
                      int(s["stat"].get("plateAppearances", 0) or 0)
                      for s in use) / pa
            teams = [s["team"]["name"] for s in splits if "team" in s]
            out[p["id"]] = {"ops": ops, "pa": pa, "teams": teams}
    return out


def team_hitting(team_id, season=None):
    season = season or dt.date.today().year
    def fetch():
        data = _get(f"teams/{team_id}/stats", {
            "group": "hitting", "stats": "season", "season": season,
            "fields": "stats,splits,stat,ops,plateAppearances,runs"})
        try:
            s = data["stats"][0]["splits"][0]["stat"]
            return {"ops": float(s["ops"]), "pa": s.get("plateAppearances"),
                    "runs": s.get("runs")}
        except (KeyError, IndexError, ValueError):
            return None
    return _cached("teamhit", (team_id, season), fetch)


def roster_strength(team_id, team_name=None, season=None):
    """Compare the CURRENT roster's hitting strength (stats travel with the
    player, so trades count immediately) to the strength embedded in the
    team's season-to-date numbers. Positive delta => lineup is now better
    than the one that produced the season stats."""
    def build():
        roster = team_roster(team_id)
        hitters = [r for r in roster if r["pos"] != "P"]
        stats = players_hitting([h["id"] for h in hitters], season)
        w_sum = ops_sum = 0.0
        additions = []
        for h in hitters:
            s = stats.get(h["id"])
            if not s or s["pa"] < MIN_PA:
                continue
            shrunk = (s["ops"] * s["pa"] + LEAGUE_AVG_OPS * OPS_SHRINK_PA) / (s["pa"] + OPS_SHRINK_PA)
            ops_sum += shrunk * s["pa"]
            w_sum += s["pa"]
            if team_name and s["teams"] and all(t != team_name for t in s["teams"]) \
                    and s["pa"] >= 100:
                additions.append((h["name"], s["ops"], s["teams"][-1]))
            elif team_name and len(s["teams"]) > 1 and s["pa"] >= 100 \
                    and s["teams"][-1] == team_name:
                additions.append((h["name"], s["ops"], s["teams"][0]))
        if w_sum <= 0:
            return None
        ops_now = ops_sum / w_sum
        th = team_hitting(team_id, season)
        if not th:
            return None
        delta_runs = OPS_TO_RUNS * (ops_now - th["ops"])
        return {"ops_now": round(ops_now, 3), "ops_season": th["ops"],
                "delta_runs_pg": round(delta_runs, 2),
                "additions": sorted(additions, key=lambda a: -a[1])[:3]}
    return _cached("strength", (team_id, season), build)


def apply_roster_overlay(p_home, home_strength, away_strength):
    """Shift home win probability for current-roster vs season-embedded lineup
    quality (captures trades/roster moves the day they happen). Capped +/-8 pts."""
    import math
    if not home_strength or not away_strength:
        return p_home, 0.0
    net = home_strength["delta_runs_pg"] - away_strength["delta_runs_pg"]
    logit = math.log(p_home / (1 - p_home)) + ROSTER_LOGIT_PER_RUN * net
    p_new = 1 / (1 + math.exp(-logit))
    shift = max(-ROSTER_CAP, min(ROSTER_CAP, p_new - p_home))
    return p_home + shift, shift


def apply_pitcher_overlay(p_home, home_line, away_line):
    """Shift the model's home win probability for the announced starter gap.

    Uses shrunk season ERA (regressed to league average by innings pitched) so a
    hot 3-start sample doesn't swing the number. The logit shift per run of ERA
    gap (0.18) reflects a starter covering ~60% of the game with ~40% regression
    on season-to-date ERA as a forecast of true talent. Capped at +/-12 prob pts.
    """
    import math
    if not home_line or not away_line:
        return p_home, 0.0
    gap = away_line["era_shrunk"] - home_line["era_shrunk"]  # positive favors home
    logit = math.log(p_home / (1 - p_home)) + OVERLAY_LOGIT_PER_RUN * gap
    p_new = 1 / (1 + math.exp(-logit))
    shift = max(-OVERLAY_CAP, min(OVERLAY_CAP, p_new - p_home))
    return p_home + shift, shift
