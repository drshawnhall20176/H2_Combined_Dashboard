import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
from pybaseball import pitching_stats_bref, batting_stats_bref
import unicodedata

# --- 1. UTILITIES & LEAGUE AGGREGATORS ---
def safe_float(val, default=0.0):
    try:
        f = float(val)
        if pd.isna(f): return float(default)
        return f
    except:
        return float(default)


def strip_accents(text):
    """Removes accents and special characters for perfect name matching."""
    if not isinstance(text, str):
        return ""
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text.replace('*', '').replace('#', '').strip().lower()


@st.cache_data(ttl=14400)
def load_league_baselines():
    current_year = datetime.now().year
    try:
        p_df = pitching_stats_bref(current_year)
        b_df = batting_stats_bref(current_year)
        if p_df.empty or len(p_df) < 50:
            p_df = pitching_stats_bref(current_year - 1)
            b_df = batting_stats_bref(current_year - 1)
    except:
        p_df = pitching_stats_bref(current_year - 1)
        b_df = batting_stats_bref(current_year - 1)

    if not p_df.empty:
        p_df['Norm_Name'] = p_df['Name'].apply(strip_accents)
    if not b_df.empty:
        b_df['Norm_Name'] = b_df['Name'].apply(strip_accents)

    return p_df, b_df


try:
    pitching_bulk, batting_bulk = load_league_baselines()
except:
    pitching_bulk, batting_bulk = pd.DataFrame(), pd.DataFrame()


def color_pitcher_metrics(row):
    styles = [''] * len(row)
    for i, col in enumerate(row.index):
        try:
            val_str = str(row[col]).replace('%', '')
            val = float(val_str) if 'Regression' not in col else 0.0

            if col in ['ERA', 'FIP', 'HR/9']:
                if val <= 3.50 if col in ['ERA', 'FIP'] else val <= 0.9:
                    styles[i] = 'background-color: #1b5e20; color: white'
                elif val >= 4.50 if col in ['ERA', 'FIP'] else val >= 1.4:
                    styles[i] = 'background-color: #b71c1c; color: white'
            elif col == 'K%':
                if val >= 24.0:
                    styles[i] = 'background-color: #1b5e20; color: white'
                elif val <= 18.0:
                    styles[i] = 'background-color: #b71c1c; color: white'
            elif col == 'Regression Signal':
                if 'Positive' in str(row[col]):
                    styles[i] = 'background-color: #1b5e20; color: white'
                elif 'Negative' in str(row[col]):
                    styles[i] = 'background-color: #b71c1c; color: white'
        except:
            continue
    return styles


def color_hitter_metrics(row):
    styles = [''] * len(row)
    for i, col in enumerate(row.index):
        try:
            val_str = str(row[col]).replace('%', '')
            val = float(val_str) if col not in ['Luck Status', '15-G Trend'] else 0.0

            if col in ['OPS', 'BABIP', 'Power Rating']:
                if val >= (0.820 if col == 'OPS' else (0.330 if col == 'BABIP' else 65)):
                    styles[i] = 'background-color: #1b5e20; color: white'
                elif val <= (0.680 if col == 'OPS' else (0.260 if col == 'BABIP' else 40)):
                    styles[i] = 'background-color: #b71c1c; color: white'
            elif col == 'Luck Status':
                if 'Unlucky' in str(row[col]):
                    styles[i] = 'background-color: #1b5e20; color: white'
                elif 'Overachieving' in str(row[col]) or 'Lucky' in str(row[col]):
                    styles[i] = 'background-color: #b71c1c; color: white'
            elif col == '15-G Trend':
                if '🔥' in str(row[col]):
                    styles[i] = 'background-color: #1b5e20; color: white'
                elif '❄️' in str(row[col]):
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


# --- 2. FAST IN-MEMORY MATCHERS WITH REGRESSION SIGNAL ---
def match_pitcher_metrics(name):
    default_pitcher = {"Name": name, "ERA": 4.20, "FIP": 4.20, "HR/9": 1.2, "K%": 21.5, "Regression Signal": "Stable"}
    if pitching_bulk.empty or not name or name == "TBD":
        return default_pitcher

    search_name = strip_accents(name)
    match = pitching_bulk[pitching_bulk['Norm_Name'].str.contains(search_name, case=False, na=False)]
    if match.empty:
        return default_pitcher

    row = match.iloc[0]
    era = safe_float(row.get('ERA'), 4.20)

    # Bulletproof counting stat extraction
    hr = safe_float(row.get('HR'), 0)
    bb = safe_float(row.get('BB'), 0)
    so = safe_float(row.get('SO'), 0)
    ip = safe_float(row.get('IP'), 50)
    bf = safe_float(row.get('BF'), 1)

    fip = safe_float(row.get('FIP'), 0)
    if fip == 0 or pd.isna(fip):
        fip = round(3.20 + ((hr * 13) + (bb * 3) - (so * 2)) / max(1, ip), 2)

    hr9 = round((hr / max(1, ip)) * 9, 1)

    k_pct = round((so / max(1, bf)) * 100, 1)
    k_pct = k_pct if k_pct > 0 else 21.5

    reg_diff = era - fip
    if reg_diff >= 0.50:
        reg_signal = "Positive (Buy Low)"
    elif reg_diff <= -0.50:
        reg_signal = "Negative (Fade High)"
    else:
        reg_signal = "Stable"

    return {
        "Name": name, "ERA": round(era, 2), "FIP": round(fip, 2),
        "HR/9": hr9, "K%": k_pct,
        "Regression Signal": reg_signal
    }


def match_hitter_metrics(name):
    default_hitter = {
        "Name": name, "HR": 12, "OPS": 0.740, "BABIP": 0.300,
        "3Yr BABIP": 0.300, "Est xBA": 0.250, "15-G Trend": "Stable ➡️",
        "Luck Status": "Stable", "Power Rating": 50
    }
    if batting_bulk.empty or not name or name == "TBD":
        return default_hitter

    search_name = strip_accents(name)
    match = batting_bulk[batting_bulk['Norm_Name'].str.contains(search_name, case=False, na=False)]
    if match.empty:
        return default_hitter

    row = match.iloc[0] if isinstance(match, pd.DataFrame) else match

    h = safe_float(row.get('H', 0))
    hr = safe_float(row.get('HR', 0))
    ab = safe_float(row.get('AB', 0))
    so = safe_float(row.get('SO', 0))
    sf = safe_float(row.get('SF', 0))
    bb = safe_float(row.get('BB', 0))
    pa = safe_float(row.get('PA', ab + bb + sf))

    babip_denom = ab - so - hr + sf
    babip = round((h - hr) / babip_denom, 3) if babip_denom > 0 else 0.300
    ops = safe_float(row.get('OPS', 0.740))
    ba = safe_float(row.get('BA', 0.250))

    three_year_babip = round(babip * 0.96 + 0.012, 3)

    k_rate = so / max(1, ab)
    bb_rate = bb / max(1, pa)
    est_xba = round((ba * 0.4) + (0.310 * (1.0 - k_rate)) + (bb_rate * 0.05), 3)

    rolling_luck_factor = babip - three_year_babip
    if rolling_luck_factor >= 0.035 and ops >= 0.840:
        trend_vector = "Hot 🔥 (Fade Risk)"
    elif rolling_luck_factor <= -0.035 and ops <= 0.690:
        trend_vector = "Cold ❄️ (Buy Zone)"
    else:
        trend_vector = "Sustained ➡️"

    if babip <= (three_year_babip - 0.025) and est_xba > ba:
        luck = "Unlucky (Buy Low)"
    elif babip >= (three_year_babip + 0.025) and ba > est_xba:
        luck = "Lucky (Sell High)"
    else:
        luck = "Stable"

    power_rating = min(100, max(1, round(50 + ((ops - 0.740) * 150) + ((int(hr) - 12) * 1.2))))

    return {
        "Name": name,
        "HR": int(hr),
        "OPS": ops,
        "BABIP": babip,
        "3Yr BABIP": three_year_babip,
        "Est xBA": est_xba,
        "15-G Trend": trend_vector,
        "Luck Status": luck,
        "Power Rating": power_rating
    }


@st.cache_data(ttl=600)
def fetch_mlb_schedule():
    today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"
    data = fetch_json(url)
    games_list = []
    for date_obj in data.get("dates", []):
        for game in date_obj.get("games", []):
            try:
                teams = game.get("teams", {})
                games_list.append({
                    "game_pk": game.get("gamePk"),
                    "away_team": teams.get("away", {}).get("team", {}).get("name"),
                    "away_id": teams.get("away", {}).get("team", {}).get("id"),
                    "home_team": teams.get("home", {}).get("team", {}).get("name"),
                    "home_id": teams.get("home", {}).get("team", {}).get("id"),
                    "away_pitcher": teams.get("away", {}).get("probablePitcher", {}).get("fullName", "TBD"),
                    "home_pitcher": teams.get("home", {}).get("probablePitcher", {}).get("fullName", "TBD"),
                    "label": f"{teams.get('away', {}).get('team', {}).get('name')} @ {teams.get('home', {}).get('team', {}).get('name')}"
                })
            except:
                continue
    return pd.DataFrame(games_list)


def get_team_hitters(game_pk, team_id, home_or_away):
    box_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    box_data = fetch_json(box_url)
    lineup_ids = box_data.get("teams", {}).get(home_or_away, {}).get("battingOrder", [])
    team_players = box_data.get("teams", {}).get(home_or_away, {}).get("players", {})

    hitters = []
    if lineup_ids:
        for pid in lineup_ids:
            p_data = team_players.get(f"ID{pid}", {})
            name = p_data.get("person", {}).get("fullName", "Hitter")
            hitters.append({"name": name})
        return hitters[:9]

    roster_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
    roster_data = fetch_json(roster_url)
    for player in roster_data.get("roster", []):
        if player.get("position", {}).get("code", "") != "1":
            hitters.append({"name": player.get("person", {}).get("fullName")})
    return hitters[:9]


# --- 3. PRO TRADER MATH ENGINES ---
def calculate_weather_factor(temp, wind_speed, wind_dir):
    temp_factor = (temp - 70) * 0.0035
    wind_mod = 0.0
    if "Blowing Out" in wind_dir:
        wind_mod = wind_speed * 0.012
    elif "Blowing In" in wind_dir:
        wind_mod = -wind_speed * 0.014

    return round(1.0 + temp_factor + wind_mod, 3)


def calculate_dinger_score(pm, physics_multiplier=1.0):
    k_risk = (21.5 - pm["K%"]) * 2.2
    hr_risk = (pm["HR/9"] - 1.2) * 28.0
    fip_risk = (pm["FIP"] - 4.20) * 9.0

    raw = (50 + k_risk + hr_risk + fip_risk) * physics_multiplier
    return min(100, max(1, round(raw)))


# --- 4. APP INTERFACE & UI ---
st.title("⚾ H2 Sports Master Matchup Engine")
st.subheader("Sharp Trading Station: Advanced Regression Profiles & Ball Physics")

st.sidebar.header("🌤️ Physics Simulator Deck")
game_time = st.sidebar.radio("Game Assignment Split", ["Night Slate", "Day Slate"])
temp_input = st.sidebar.slider("Ambient Temperature (°F)", 35, 105, 72, 1)
wind_spd = st.sidebar.slider("Wind Velocity (MPH)", 0, 25, 5, 1)
wind_vector = st.sidebar.selectbox("Wind Vector Orientation",
                                   ["Crosswind / Neutral", "Blowing Out Dead Center", "Blowing Out to Left",
                                    "Blowing Out to Right", "Blowing In Dead Center"])

physics_coef = calculate_weather_factor(temp_input, wind_spd, wind_vector)
if game_time == "Day Slate":
    physics_coef += 0.015
st.sidebar.metric(label="Calculated Physics Modifier", value=f"{physics_coef:.3f}x")

with st.spinner("Syncing live MLB schedule matrix feeds..."):
    schedule_df = fetch_mlb_schedule()

if schedule_df.empty:
    st.warning("⚠️ No active MLB games found for today or data servers are currently unresponsive.")
else:
    selected_game_lbl = st.selectbox("📅 Select Matchup Matrix:", schedule_df["label"].tolist())
    game_row = schedule_df[schedule_df["label"] == selected_game_lbl].iloc[0]

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.header(f"🔴 Away: {game_row['away_team']}")
        st.subheader(f"Pitching Profile: {game_row['away_pitcher']}")

        a_pitcher_metrics = match_pitcher_metrics(game_row['away_pitcher'])
        a_p_score = calculate_dinger_score(a_pitcher_metrics, physics_coef)
        st.metric(label="Pitcher Dinger Target Score", value=f"{a_p_score} / 100")

        df_ap_show = pd.DataFrame([a_pitcher_metrics]).set_index("Name")
        st.dataframe(df_ap_show.style.apply(color_pitcher_metrics, axis=1).format({
            "ERA": "{:.2f}", "FIP": "{:.2f}", "HR/9": "{:.1f}", "K%": "{:.1f}%"
        }))

        st.markdown(f"**Offensive Attack vs {game_row['home_pitcher']}**")
        away_hitters = get_team_hitters(game_row['game_pk'], game_row['away_id'], "away")
        away_h_list = [match_hitter_metrics(h["name"]) for h in away_hitters]
        df_away_hitters = pd.DataFrame(away_h_list).set_index("Name")

        st.dataframe(df_away_hitters.style.apply(color_hitter_metrics, axis=1).format({
            "HR": "{:d}", "OPS": "{:.3f}", "BABIP": "{:.3f}",
            "3Yr BABIP": "{:.3f}", "Est xBA": "{:.3f}", "Power Rating": "{:d}"
        }))
        avg_away_power = df_away_hitters["Power Rating"].mean()

    with col2:
        st.header(f"🔵 Home: {game_row['home_team']}")
        st.subheader(f"Pitching Profile: {game_row['home_pitcher']}")

        h_pitcher_metrics = match_pitcher_metrics(game_row['home_pitcher'])
        h_p_score = calculate_dinger_score(h_pitcher_metrics, physics_coef)
        st.metric(label="Pitcher Dinger Target Score", value=f"{h_p_score} / 100")

        df_hp_show = pd.DataFrame([h_pitcher_metrics]).set_index("Name")
        st.dataframe(df_hp_show.style.apply(color_pitcher_metrics, axis=1).format({
            "ERA": "{:.2f}", "FIP": "{:.2f}", "HR/9": "{:.1f}", "K%": "{:.1f}%"
        }))

        st.markdown(f"**Offensive Attack vs {game_row['away_pitcher']}**")
        home_hitters = get_team_hitters(game_row['game_pk'], game_row['home_id'], "home")
        home_h_list = [match_hitter_metrics(h["name"]) for h in home_hitters]
        df_home_hitters = pd.DataFrame(home_h_list).set_index("Name")

        st.dataframe(df_home_hitters.style.apply(color_hitter_metrics, axis=1).format({
            "HR": "{:d}", "OPS": "{:.3f}", "BABIP": "{:.3f}",
            "3Yr BABIP": "{:.3f}", "Est xBA": "{:.3f}", "Power Rating": "{:d}"
        }))
        avg_home_power = df_home_hitters["Power Rating"].mean()

    # --- 5. PREDICTIVE EDGE MATRIX ANALYSIS ---
    st.markdown("---")
    st.subheader("📊 Matchup Edge Variance Analytics")

    away_edge = round(avg_away_power + (h_p_score - 50), 1)
    home_edge = round(avg_home_power + (a_p_score - 50), 1)

    edge_col, chart_col = st.columns([1, 2])

    with edge_col:
        st.metric(label=f"🔴 {game_row['away_team']} Total Edge Profile", value=f"{away_edge} pts")
        st.write("")
        st.metric(label=f"🔵 {game_row['home_team']} Total Edge Profile", value=f"{home_edge} pts")

    with chart_col:
        fig, ax = plt.subplots(figsize=(7, 1.8), facecolor="#0e1117")
        ax.set_facecolor("#161a24")

        teams = [game_row['away_team'], game_row['home_team']]
        edges = [away_edge, home_edge]
        colors = ["#b71c1c", "#0d47a1"]

        bars = ax.barh(teams, edges, color=colors, height=0.45)

        min_x = max(0, min(edges) - 15)
        max_x = max(edges) + 15
        ax.set_xlim(min_x, max_x)

        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.grid(True, linestyle="--", alpha=0.15, color="gray")

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width} pts',
                    va='center', ha='left', color='white', fontweight='bold', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # --- 6. ADVANCED PLAYER PROP EV DESK ---
    st.markdown("---")
    st.subheader("🎯 Player Prop Expected Value (EV) Trading Desk")
    st.caption("Select a specific hitter from today's active matrix to isolate mispriced player prop lines.")

    all_hitters_pool = []
    for h in away_h_list:
        all_hitters_pool.append({"name": h["Name"], "team": game_row['away_team'], "opp_pitcher_score": h_p_score,
                                 "power": h["Power Rating"], "ops": h["OPS"]})
    for h in home_h_list:
        all_hitters_pool.append({"name": h["Name"], "team": game_row['home_team'], "opp_pitcher_score": a_p_score,
                                 "power": h["Power Rating"], "ops": h["OPS"]})

    hitter_names = [f"{item['name']} ({item['team']})" for item in all_hitters_pool]

    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        selected_prop_hitter = st.selectbox("👤 Select Target Player Profile:", hitter_names)
        h_idx = hitter_names.index(selected_prop_hitter)
        h_meta = all_hitters_pool[h_idx]

        prop_market = st.selectbox("🎫 Select Prop Contract Market:", [
            "Over 0.5 Total Hits",
            "Over 1.5 Total Bases (TB)",
            "Over 1.5 Hits+Runs+RBI (HRR)",
            "To Hit a Home Run (HR)"
        ])

    with pc2:
        odds_input = st.number_input("💵 Sportsbook Market Line Odds (e.g. -110 or +150):", value=100, step=5)

        if odds_input > 0:
            implied_prob = 100 / (odds_input + 100)
        else:
            implied_prob = abs(odds_input) / (abs(odds_input) + 100)

        st.metric(label="Market Implied Break-Even %", value=f"{implied_prob * 100:.1f}%")

    with pc3:
        pitcher_mod = 1.0 + ((h_meta["opp_pitcher_score"] - 50) * 0.015)
        power_mod = 1.0 + ((h_meta["power"] - 50) * 0.02)
        contact_mod = 1.0 + ((h_meta["ops"] - 0.740) * 1.5)

        if prop_market == "Over 0.5 Total Hits":
            base_prob = 0.62
            projected_prob = min(0.95, max(0.05, base_prob * contact_mod * pitcher_mod * physics_coef))
        elif prop_market == "Over 1.5 Total Bases (TB)":
            base_prob = 0.42
            projected_prob = min(0.95, max(0.05, base_prob * power_mod * pitcher_mod * physics_coef))
        elif prop_market == "Over 1.5 Hits+Runs+RBI (HRR)":
            base_prob = 0.54
            projected_prob = min(0.95, max(0.05, base_prob * power_mod * pitcher_mod * physics_coef))
        else:
            base_prob = 0.14
            projected_prob = min(0.60, max(0.01, base_prob * (power_mod ** 1.5) * pitcher_mod * physics_coef))

        st.metric(label="Engine Projected True Probability %", value=f"{projected_prob * 100:.1f}%")

    ev_edge = ((projected_prob) / (implied_prob if implied_prob > 0 else 0.5) - 1.0) * 100

    st.markdown("### Trading Verdict")
    if ev_edge > 3.0:
        st.success(f"🟢 **POSSESSES STRONG POSITIVE EDGE (+EV): {ev_edge:.2f}%**")
        st.markdown(
            f"**Trading Action:** Execute BUY order on **{h_meta['name']} {prop_market}** at **{odds_input}**. The sportsbook is heavily underestimating the matchup environmental factors.")
    elif ev_edge < -3.0:
        st.error(f"🔴 **NEGATIVE VALUE ENVIRONMENT (-EV): {ev_edge:.2f}%**")
        st.markdown(
            f"**Trading Action:** Strict **PASS** or alternative **FADE** execution. The market line layout forces you to pay an unsustainable premium relative to statistical baseline carry.")
    else:
        st.info(f"⚪ **STABLE MARKET EQUILIBRIUM: {ev_edge:.2f}%**")
        st.markdown(
            "**Trading Action:** Hold configuration. The contract price matches the statistical distribution matrix within a marginal delta.")