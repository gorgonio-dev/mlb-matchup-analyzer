"""Sportsbook odds math: implied probability, vig removal, edge, EV, Kelly.

All functions are pure so they can be unit-tested exactly.
American odds convention: -150 means bet 150 to win 100; +130 means bet 100 to win 130.
"""

MIN_EDGE = 0.02          # minimum edge vs fair probability to call something a value bet
KELLY_FRACTION = 0.25    # quarter-Kelly: standard professional risk reduction


def american_to_decimal(a):
    a = float(a)
    if a == 0 or -100 < a < 100:
        raise ValueError("American odds must be <= -100 or >= +100")
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def american_to_implied(a):
    """Implied probability including the book's margin."""
    a = float(a)
    if a == 0 or -100 < a < 100:
        raise ValueError("American odds must be <= -100 or >= +100")
    return (-a) / (-a + 100.0) if a < 0 else 100.0 / (a + 100.0)


def novig_two_way(a1, a2):
    """Remove the vig from a two-way market by proportional normalization.
    Returns (fair_p1, fair_p2, overround)."""
    i1, i2 = american_to_implied(a1), american_to_implied(a2)
    total = i1 + i2
    return i1 / total, i2 / total, total - 1.0


def kelly_fraction(p, a):
    """Full-Kelly fraction of bankroll for a bet at American odds `a`
    with true win probability p. Returns 0 if no positive expectation."""
    b = american_to_decimal(a) - 1.0
    f = (p * b - (1.0 - p)) / b
    return max(0.0, f)


def evaluate(p_model_1, ml_team1, ml_team2=None, bankroll=None):
    """Assess both sides of a matchup given the model's probability that
    team 1 wins and the sportsbook moneyline(s).

    Returns a dict per side: implied, fair (if both lines given), edge vs fair
    (or vs implied when only one line is available), EV per $1, Kelly stake.
    """
    res = {"has_fair": ml_team2 is not None}
    if ml_team2 is not None:
        f1, f2, over = novig_two_way(ml_team1, ml_team2)
        res["overround"] = over
    else:
        f1 = american_to_implied(ml_team1)
        f2 = None

    for side, p_model, ml, fair in (
            ("team1", p_model_1, ml_team1, f1),
            ("team2", 1 - p_model_1, ml_team2, f2)):
        if ml is None:
            continue
        dec = american_to_decimal(ml)
        edge = p_model - fair
        ev = p_model * (dec - 1.0) - (1.0 - p_model)   # per $1 staked
        kf = kelly_fraction(p_model, ml)
        entry = {
            "ml": float(ml), "decimal": round(dec, 4),
            "implied": round(american_to_implied(ml), 4),
            "fair": round(fair, 4),
            "edge": round(edge, 4),
            "ev_per_dollar": round(ev, 4),
            "kelly_full": round(kf, 4),
            "kelly_quarter": round(kf * KELLY_FRACTION, 4),
            "value_bet": bool(edge >= MIN_EDGE and ev > 0),
        }
        if bankroll:
            entry["stake_quarter_kelly"] = round(bankroll * kf * KELLY_FRACTION, 2)
        res[side] = entry
    return res
