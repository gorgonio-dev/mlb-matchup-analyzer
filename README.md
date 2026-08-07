# MLB Matchup Analyzer — Web

Win-probability model for MLB matchups with live overlays (announced starters,
bullpen quality, current-roster strength) and sportsbook value analysis
(no-vig fair odds, edge, EV, quarter-Kelly staking).

Runs as a single Python process — no framework, no database server:

    pip install -r requirements.txt
    python server.py          # http://localhost:8000

The model retrains itself from the bundled game database, and a background
thread pulls new results from the MLB Stats API every 6 hours.

Deployed on Render free tier via `render.yaml`.
