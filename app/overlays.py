"""Shared live-overlay pipeline: starter + bullpen + roster adjustments.

Used by both the single-matchup view and the slate board so the numbers
always agree. All network failures degrade to partial results.
"""


def compute_live(id1, id2, n1, n2, p_team1, pitchers=None, want_vs=True,
                 venue="team1"):
    """Apply all live overlays to the base team-level probability.

    pitchers: optional pre-fetched {'home_id','away_id','home','away'} dict
    (the slate board already has starters from the schedule call).
    Returns {'p_adj', 'total_shift', 'lines', 'ok'} — team-1 perspective.
    Raises live.LiveDataError only if nothing at all could be fetched.
    """
    import live

    lines = []
    p_t1 = p_team1
    total = 0.0
    any_ok = False

    # ---- starting pitchers ----
    pp = pitchers
    if pp is None:
        try:
            pp = live.probable_pitchers(id1, id2)
        except Exception:
            pp = None
    if pp and (pp.get("home") or pp.get("away")):
        any_ok = True
        hl = al = None
        try:
            hl = live.pitcher_line(pp["home"]["id"]) if pp["home"] else None
            al = live.pitcher_line(pp["away"]["id"]) if pp["away"] else None
        except Exception:
            pass
        import predict
        native = False
        try:
            native = predict.model_knows_starters()
        except Exception:
            pass
        if native:
            # model-trained starter feature: re-predict with the announced
            # starters instead of applying the hand-calibrated ERA overlay
            hp = pp["home"]["id"] if pp.get("home") else None
            ap = pp["away"]["id"] if pp.get("away") else None
            t1_pid = hp if pp["home_id"] == id1 else ap
            t2_pid = ap if pp["home_id"] == id1 else hp
            p_new = predict.predict_matchup(id1, id2, venue,
                                            t1_pid, t2_pid)["p_team1"]
            total += p_new - p_t1
            p_t1 = p_new
            lines.append("Starters modeled natively (pitcher form learned "
                         "from 6,500 historical games).")
        else:
            p_home = p_t1 if pp["home_id"] == id1 else 1 - p_t1
            adj_home, shift = live.apply_pitcher_overlay(p_home, hl, al)
            t1_shift = shift if pp["home_id"] == id1 else -shift
            p_t1 += t1_shift
            total += t1_shift
        for side, lab in (("home", "Home"), ("away", "Away")):
            if pp.get(side):
                ln = hl if side == "home" else al
                txt = f"{lab} starter: {pp[side]['name']}"
                if ln:
                    txt += (f" — ERA {ln['era']} (WHIP {ln['whip']}, {ln['ip']} IP, "
                            f"{ln['record']})")
                if want_vs:
                    try:
                        opp = pp["away_id"] if side == "home" else pp["home_id"]
                        vs = live.pitcher_vs_team(pp[side]["id"], opp)
                        if vs:
                            txt += (f"; opponent hits {vs['avg_against']} / "
                                    f"{vs['ops_against']} OPS off him career "
                                    f"({vs['ab']} AB, {vs['so']} K, {vs['hr']} HR)")
                    except Exception:
                        pass
                lines.append(txt)
    else:
        lines.append("No announced starters for these teams today — starter overlay skipped.")

    # ---- bullpens ----
    try:
        pen1 = live.bullpen_strength(id1)
        pen2 = live.bullpen_strength(id2)
    except Exception:
        pen1 = pen2 = None
    if pen1 and pen2:
        any_ok = True
        p_t1, pshift = live.apply_bullpen_overlay(p_t1, pen1, pen2)
        total += pshift
        lines.append(f"Bullpens — {n1}: {pen1['era_shrunk']} ERA "
                     f"({pen1['relievers']} arms, {pen1['ip']} IP) | "
                     f"{n2}: {pen2['era_shrunk']} ERA "
                     f"({pen2['relievers']} arms, {pen2['ip']} IP) "
                     f"[{pshift:+.1%}]")
    else:
        lines.append("Bullpen data unavailable — bullpen overlay skipped.")

    # ---- current rosters (trades) ----
    try:
        s1 = live.roster_strength(id1, n1)
        s2 = live.roster_strength(id2, n2)
    except Exception:
        s1 = s2 = None
    if s1 and s2:
        any_ok = True
        p_t1, rshift = live.apply_roster_overlay(p_t1, s1, s2)
        total += rshift
        for nm, s in ((n1, s1), (n2, s2)):
            seg = (f"{nm} current roster: {s['ops_now']:.3f} OPS vs "
                   f"{s['ops_season']:.3f} season ({s['delta_runs_pg']:+.2f} runs/gm)")
            if s["additions"]:
                adds = ", ".join(f"{a[0]} ({a[1]:.3f} OPS, from {a[2]})"
                                 for a in s["additions"])
                seg += f"; recent additions: {adds}"
            lines.append(seg)
    else:
        lines.append("Roster data unavailable — lineup overlay skipped.")

    if not any_ok:
        import live as _l
        raise _l.LiveDataError("no live data available")

    return {"p_adj": p_t1, "total_shift": total, "lines": lines}
