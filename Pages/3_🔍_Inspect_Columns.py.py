import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from pybaseball import statcast_pitcher

# --- 1. UTILITIES & STYLING ---
def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default


def format_rate(val):
    try:
        s = f"{float(val):.3f}"
        return s[1:] if s.startswith("0.") else s
    except:
        return ".000"


def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def color_pitcher_metrics(row):
    styles = [''] * len(row)
    for i, col in enumerate(row.index):
        val_str = str(row[col]).replace('%', '')
        try:
            val = float(val_str)
            if col in ['ERA', 'BRL%', 'HH%']:
                if val <= 3.50 if col == 'ERA' else (val <= 7.0 if col == 'BRL%' else val <= 35.0):
                    styles[i] = 'background-color: #1b5e20; color: white'
                elif val >= 4.50 if col == 'ERA' else (val >= 10.0 if col == 'BRL%' else val >= 40.0):
                    styles[i] = 'background-color: #b71c1c; color: white'
            elif col in ['K%', 'K/9']:
                if val >= 25.0 if col == 'K%' else val >= 9.0:
                    styles[i] = 'background-color: #1b5e20; color: white'
                elif val <= 18.0 if col == 'K%' else val <= 7.0:
                    styles[i] = 'background-color: #b71c1c; color: white'
        except:
            continue
    return styles


@st.cache_data(ttl=300)
def fetch_json(url, params=None):
    try:
        response = requests.get(url, params=params or {}, timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}


# --- 2. ENGINE DATA FETCHING ---
def get_pitcher_metrics(pitcher_id):
    if not pitcher_id:
        return {"Name": "TBD", "HR/9": 0.0, "K/9": 0.0, "ERA": 0.0, "WHIP": 0.0, "OBA": 0.0, "Hand": "R"}
    data = fetch_json(
        f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=[pitching],type=[season])")
    try:
        p_data = data['people'][0]
        s = p_data['stats'][0]['splits'][0]['stat']
        so, ip = float(s.get('strikeOuts', 0)), float(s.get('inningsPitched', 0))
        return {
            "Name": p_data.get('fullName', 'TBD'), "Hand": p_data.get('pitchHand', {}).get('code', 'R'),
            "HR/9": float(s.get('homeRunsPer9', 0.0)), "K/9": (so / ip * 9) if ip > 0 else 0.0,
            "ERA": float(s.get('era', 0.0)), "WHIP": float(s.get('whip', 0.0)), "OBA": float(s.get('avg', 0.0))
        }
    except:
        return {"Name": "TBD", "Hand": "R", "HR/9": 0.0, "K/9": 0.0, "ERA": 0.0, "WHIP": 0.0, "OBA": 0.0}


def get_hitter_stats(player_id, p_met, b_hand, team_name, game_label):
    data = fetch_json(
        f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=stats(group=[hitting],type=[season])")
    try:
        p_data = data['people'][0]
        s = p_data['stats'][0]['splits'][0]['stat']
        hr, slg, avg, ops = float(s.get('homeRuns', 0)), float(s.get('slg', 0)), float(s.get('avg', 0)), float(
            s.get('ops', 0))
        iso, k_pct = slg - avg, (float(s.get('strikeOuts', 0)) / max(float(s.get('plateAppearances', 1)), 1))
        matchup = (float(s.get('wOBA', 0)) * 500) + (iso * 1000)
        return {
            "Hitter": p_data['fullName'], "Team": team_name, "GameLabel": game_label, "Hand": b_hand,
            "Opp Pitcher": p_met.get('Name', 'TBD'), "Opp Pitcher Hand": p_met.get('Hand', 'R'),
            "Advantage": "Advantage" if b_hand != p_met.get('Hand') else "Disadvantage",
            "HR": hr, "Hits": float(s.get('hits', 0)), "TB": float(s.get('totalBases', 0)),
            "AVG": avg, "OPS": ops, "SLG": slg, "ISO": iso, "K%": k_pct, "Matchup": matchup
        }
    except:
        return None


def style_df(data):
    cols = ["Hitter", "Team", "Hand", "Opp Pitcher", "Opp Pitcher Hand", "Advantage", "HR", "Hits", "TB", "AVG", "OPS",
            "SLG", "ISO", "K%", "Matchup"]
    display_df = data[[c for c in cols if c in data.columns]]
    # Advantage coloring removed below
    return display_df.style.format({
        "HR": "{:.1f}", "Hits": "{:.1f}", "TB": "{:.1f}", "AVG": "{:.3f}",
        "OPS": "{:.3f}", "SLG": "{:.3f}", "ISO": "{:.3f}", "Matchup": "{:.1f}", "K%": "{:.1%}"
    }).background_gradient(cmap="RdYlGn", subset=["HR", "Hits", "TB", "AVG", "OPS", "SLG", "ISO", "Matchup"]) \
        .background_gradient(cmap="RdYlGn_r", subset=["K%"])


# --- MAIN EXECUTION ---
eastern = pytz.timezone("US/Eastern")
today = datetime.now(eastern).strftime('%Y-%m-%d')
sched = fetch_json("https://statsapi.mlb.com/api/v1/schedule",
                   {"sportId": 1, "date": today, "hydrate": "probablePitcher"})

if st.button("🔄 Refresh Data"): st.cache_data.clear(); st.rerun()
st.title("⚾ H2 Sports Master Dinger Engine")

if 'dates' in sched and len(sched['dates']) > 0:
    master_list, game_meta = [], []
    with st.spinner("Compiling Telemetry..."):
        for game in sorted(sched['dates'][0]['games'], key=lambda x: x.get('gameDate', '')):
            game_label = f"{game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']} (Game {game.get('gameNumber', 1)})"
            h_met = get_pitcher_metrics(game['teams']['home'].get('probablePitcher', {}).get('id'))
            a_met = get_pitcher_metrics(game['teams']['away'].get('probablePitcher', {}).get('id'))
            game_meta.append(
                {"Label": game_label, "Home": h_met, "Away": a_met, "AwayName": game['teams']['away']['team']['name'],
                 "HomeName": game['teams']['home']['team']['name']})

            boxscore = fetch_json(f"https://statsapi.mlb.com/api/v1/game/{game['gamePk']}/boxscore")
            use_roster = not (boxscore and 'teams' in boxscore and boxscore['teams']['home'].get('battingOrder'))

            for team_key, p_met in [('away', h_met), ('home', a_met)]:
                team_id = game['teams'][team_key]['team']['id']
                starters = boxscore['teams'][team_key].get('battingOrder', []) if not use_roster else [p['person']['id']
                                                                                                       for p in
                                                                                                       fetch_json(
                                                                                                           f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/Active?hydrate=person").get(
                                                                                                           'roster', [])
                                                                                                       if
                                                                                                       p.get('position',
                                                                                                             {}).get(
                                                                                                           'abbreviation') != 'P']
                for pid in starters:
                    p_info = fetch_json(f"https://statsapi.mlb.com/api/v1/people/{pid}")
                    b_hand = p_info['people'][0].get('batSide', {}).get('code', 'R') if (
                                p_info and 'people' in p_info) else 'R'
                    h = get_hitter_stats(pid, p_met, b_hand, game['teams'][team_key]['team']['name'], game_label)
                    if h: master_list.append(h)

    df_all = pd.DataFrame(master_list)

    # Leaderboards
    c1, c2, c3 = st.columns(3)
    c1.subheader("Top Matchups");
    c1.dataframe(df_all.sort_values("Matchup", ascending=False).head(5).round(1), hide_index=True)
    c2.subheader("Advantage Hitters");
    c2.dataframe(df_all[df_all['Advantage'] == 'Advantage'].nlargest(5, "Matchup").round(1), hide_index=True)
    c3.subheader("High ISO Leaders");
    c3.dataframe(df_all.nlargest(5, "ISO").round(1), hide_index=True)

    # Expanders
    for meta in game_meta:
        with st.expander(meta['Label']):
            st.markdown(
                f"✈️ **Away ({meta['Away']['Name']}):** K/9: {meta['Away']['K/9']:.1f} | ERA: {meta['Away']['ERA']:.2f} | WHIP: {meta['Away']['WHIP']:.2f}")
            st.markdown(
                f"🏠 **Home ({meta['Home']['Name']}):** K/9: {meta['Home']['K/9']:.1f} | ERA: {meta['Home']['ERA']:.2f} | WHIP: {meta['Home']['WHIP']:.2f}")
            t1, t2 = st.tabs(["✈️ Away Offense", "🏠 Home Offense"])
            with t1: st.dataframe(
                style_df(df_all[(df_all['GameLabel'] == meta['Label']) & (df_all['Team'] == meta['AwayName'])]),
                use_container_width=True)
            with t2: st.dataframe(
                style_df(df_all[(df_all['GameLabel'] == meta['Label']) & (df_all['Team'] == meta['HomeName'])]),
                use_container_width=True)