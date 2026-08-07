"""
Leak-free feature engineering for MLB win-probability model.

Replays all games in chronological order, maintaining running team state
(Elo, season form, venue splits, head-to-head). Every feature attached to a
game reflects ONLY information available before that game was played.
"""
import sqlite3
from collections import defaultdict, deque

# ---- tunables ----
ELO_K = 4.0
ELO_SEASON_REGRESS = 1 / 3        # pull toward 1500 between seasons
SHRINK_N = 25                      # games of shrinkage toward prior for season rates
H2H_SHRINK = 8                     # pseudo-meetings of 0.5 for head-to-head
H2H_WINDOW_DAYS = 730              # ~2 years
REST_CAP = 3

FEATURES = [
    "elo_diff",          # home Elo - away Elo
    "wpct_diff",         # shrunk season win% diff
    "rundiff_pg_diff",   # shrunk season run differential per game diff
    "last10_diff",       # last-10-games win% diff
    "venue_form_diff",   # home team's home win% - away team's road win%
    "h2h_edge",          # home team's shrunk win% vs this opponent (2-yr window) - 0.5
    "rest_diff",         # rest-day diff, capped
    "starter_ra_diff",   # away starter's shrunk runs-allowed/start - home's (leak-free)
]

# starting-pitcher rolling quality
SP_LEAGUE_RA = 4.6        # league-average team runs allowed per game
SP_SHRINK_STARTS = 10.0   # starts of league-average shrinkage
SP_SEASON_DECAY = 0.6     # carry-over weight across seasons


class PitcherState:
    __slots__ = ("n", "ra", "season")

    def __init__(self):
        self.n = 0.0
        self.ra = 0.0
        self.season = None

    def roll_season(self, season):
        if self.season is not None and season != self.season:
            self.n *= SP_SEASON_DECAY
            self.ra *= SP_SEASON_DECAY
        self.season = season

    def shrunk_ra(self):
        return (self.ra + SP_SHRINK_STARTS * SP_LEAGUE_RA) / (self.n + SP_SHRINK_STARTS)


class TeamState:
    def __init__(self):
        self.elo = 1500.0
        self.season = None
        self.prior_wpct = 0.5
        self.prior_rdpg = 0.0
        self.reset_season()

    def reset_season(self):
        self.w = 0
        self.l = 0
        self.rf = 0
        self.ra = 0
        self.last10 = deque(maxlen=10)
        self.home_w = 0
        self.home_l = 0
        self.away_w = 0
        self.away_l = 0
        self.last_date = None

    @property
    def n(self):
        return self.w + self.l

    def shrunk_wpct(self):
        return (self.w + SHRINK_N * self.prior_wpct) / (self.n + SHRINK_N)

    def shrunk_rdpg(self):
        rd = self.rf - self.ra
        return (rd + SHRINK_N * self.prior_rdpg) / (self.n + SHRINK_N)

    def last10_wpct(self):
        if not self.last10:
            return self.shrunk_wpct()
        # small shrink so a 3-game sample isn't extreme
        return (sum(self.last10) + 3 * 0.5) / (len(self.last10) + 3)

    def venue_wpct(self, home):
        w, lo = (self.home_w, self.home_l) if home else (self.away_w, self.away_l)
        return (w + 10 * self.prior_wpct) / (w + lo + 10)


def _date_ord(datestr):
    y, m, d = int(datestr[:4]), int(datestr[5:7]), int(datestr[8:10])
    # good-enough ordinal for rest-day and window math
    return y * 372 + (m - 1) * 31 + (d - 1)


def replay(db_path, collect_rows=True):
    """Replay all games chronologically.

    Returns (rows, state) where rows is a list of dicts with features + label
    for each game, and state is the post-replay world state usable for
    predicting future matchups.
    """
    con = sqlite3.connect(db_path)
    games = con.execute(
        "SELECT gamePk, date, season, away_id, home_id, away_score, home_score, home_win "
        "FROM games ORDER BY date, gamePk").fetchall()
    try:
        starters = dict((pk, (hp, ap)) for pk, hp, ap in
                        con.execute("SELECT gamePk, home_pid, away_pid FROM starters"))
    except sqlite3.OperationalError:
        starters = {}
    con.close()

    teams = defaultdict(TeamState)
    pitchers = defaultdict(PitcherState)
    h2h = defaultdict(deque)   # (a,b) sorted tuple -> deque of (ord, winner_id)
    rows = []
    last_date = games[-1][1] if games else None

    for pk, date, season, away, home, asc, hsc, hw in games:
        th, ta = teams[home], teams[away]
        for t in (th, ta):
            if t.season != season:
                if t.season is not None:
                    n = t.n
                    if n > 0:
                        t.prior_wpct = t.w / n
                        t.prior_rdpg = (t.rf - t.ra) / n
                    t.elo = 1500.0 + (t.elo - 1500.0) * (1 - ELO_SEASON_REGRESS)
                t.season = season
                t.reset_season()

        o = _date_ord(date)
        key = (min(home, away), max(home, away))
        dq = h2h[key]
        while dq and o - dq[0][0] > H2H_WINDOW_DAYS:
            dq.popleft()

        hp_id, ap_id = starters.get(pk, (None, None))
        sp_h = pitchers[hp_id] if hp_id else None
        sp_a = pitchers[ap_id] if ap_id else None
        for sp in (sp_h, sp_a):
            if sp is not None:
                sp.roll_season(season)
        ra_h = sp_h.shrunk_ra() if sp_h is not None else SP_LEAGUE_RA
        ra_a = sp_a.shrunk_ra() if sp_a is not None else SP_LEAGUE_RA

        if collect_rows:
            hw_h2h = sum(1 for _, wnr in dq if wnr == home)
            n_h2h = len(dq)
            h2h_p = (hw_h2h + H2H_SHRINK * 0.5) / (n_h2h + H2H_SHRINK)
            rest_h = min(o - th.last_date, REST_CAP) if th.last_date else 1
            rest_a = min(o - ta.last_date, REST_CAP) if ta.last_date else 1
            rows.append({
                "gamePk": pk, "date": date, "season": season,
                "home_id": home, "away_id": away, "home_win": hw,
                "n_home": th.n, "n_away": ta.n,
                "elo_diff": th.elo - ta.elo,
                "wpct_diff": th.shrunk_wpct() - ta.shrunk_wpct(),
                "rundiff_pg_diff": th.shrunk_rdpg() - ta.shrunk_rdpg(),
                "last10_diff": th.last10_wpct() - ta.last10_wpct(),
                "venue_form_diff": th.venue_wpct(True) - ta.venue_wpct(False),
                "h2h_edge": h2h_p - 0.5,
                "rest_diff": rest_h - rest_a,
                "starter_ra_diff": ra_a - ra_h,   # positive = weaker away starter
            })

        # ---- update state with the result ----
        exp_h = 1 / (1 + 10 ** ((ta.elo - th.elo) / 400))
        margin = abs(hsc - asc)
        mult = ((margin + 1) ** 0.7) / 2.0   # margin-of-victory multiplier
        delta = ELO_K * mult * ((1 if hw else 0) - exp_h)
        th.elo += delta
        ta.elo -= delta

        th.w += hw; th.l += 1 - hw; th.rf += hsc; th.ra += asc
        ta.w += 1 - hw; ta.l += hw; ta.rf += asc; ta.ra += hsc
        th.last10.append(hw); ta.last10.append(1 - hw)
        th.home_w += hw; th.home_l += 1 - hw
        ta.away_w += 1 - hw; ta.away_l += hw
        th.last_date = o; ta.last_date = o
        dq.append((o, home if hw else away))

        # charge each starter with the runs his team allowed in his start
        if sp_h is not None:
            sp_h.n += 1; sp_h.ra += asc
        if sp_a is not None:
            sp_a.n += 1; sp_a.ra += hsc

    state = {"teams": dict(teams), "h2h": {k: list(v) for k, v in h2h.items()},
             "pitchers": dict(pitchers), "as_of": last_date}
    return rows, state


def matchup_features(state, home_id, away_id, today_ord=None,
                     home_pid=None, away_pid=None):
    """Feature vector for a hypothetical game today: away_id at home_id.
    Optional starter IDs use each pitcher's leak-free rolling quality; when a
    starter is unknown (or has no MLB history) a league-average starter is
    assumed, so the feature contributes nothing rather than guessing."""
    th = state["teams"][home_id]
    ta = state["teams"][away_id]
    pitchers = state.get("pitchers", {})

    def sp_ra(pid):
        sp = pitchers.get(pid) if pid else None
        return sp.shrunk_ra() if sp is not None and sp.n > 0 else SP_LEAGUE_RA
    ra_h, ra_a = sp_ra(home_pid), sp_ra(away_pid)
    key = (min(home_id, away_id), max(home_id, away_id))
    dq = state["h2h"].get(key, [])
    o = today_ord if today_ord is not None else _date_ord(state["as_of"]) + 1
    dq = [x for x in dq if o - x[0] <= H2H_WINDOW_DAYS]
    hw_h2h = sum(1 for _, wnr in dq if wnr == home_id)
    h2h_p = (hw_h2h + H2H_SHRINK * 0.5) / (len(dq) + H2H_SHRINK)
    return {
        "elo_diff": th.elo - ta.elo,
        "wpct_diff": th.shrunk_wpct() - ta.shrunk_wpct(),
        "rundiff_pg_diff": th.shrunk_rdpg() - ta.shrunk_rdpg(),
        "last10_diff": th.last10_wpct() - ta.last10_wpct(),
        "venue_form_diff": th.venue_wpct(True) - ta.venue_wpct(False),
        "h2h_edge": h2h_p - 0.5,
        "rest_diff": 0,
        "starter_ra_diff": ra_a - ra_h,
    }, {"h2h_n": len(dq), "h2h_home_wins": hw_h2h,
        "home_sp_ra": round(ra_h, 2), "away_sp_ra": round(ra_a, 2)}
