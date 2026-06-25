import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from pybaseball import statcast_pitcher

# --- 1. UTILITIES ---
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


@st.cache_data(ttl=300)
def fetch_json(url, params=None):
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}


# --- 2. ENGINE DATA FETCHING (STATSAPI) ---
@st.cache_data(ttl=3600)
def get_league_statcast_map():
    year = datetime.now().year
    data_adv = fetch_json(
        f"https://statsapi.mlb.com/api/v1/stats?stats=statcastAdvanced&group=pitching&playerPool=all&season={year}&gameType=R&limit=5000")
    data_exp = fetch_json(
        f"https://statsapi.mlb.com/api/v1/stats?stats=expectedStatistics&group=pitching&playerPool=all&season={year}&gameType=R&limit=5000")
    mapping = {}
    if 'stats' in data_adv:
        for group in data_adv['stats']:
            for split in group['splits']:
                p_id = str(split['player']['id'])
                if p_id not in mapping: mapping[p_id] = {}
                mapping[p_id]["BRL%"] = round(float(split['stat'].get('barrelPercentage', 0)), 1)
    if 'stats' in data_exp:
        for group in data_exp['stats']:
            for split in group['splits']:
                p_id = str(split['player']['id'])
                if p_id not in mapping: mapping[p_id] = {}
                mapping[p_id].update(
                    {"xERA": float(split['stat'].get('estimatedEra', 0)), "FIP": float(split['stat'].get('fip', 0))})
    return mapping


@st.cache_data(ttl=3600)
def get_live_probables():
    today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    data = fetch_json("https://statsapi.mlb.com/api/v1/schedule",
                      params={"sportId": 1, "date": today, "hydrate": "probablePitcher"})
    starters = []
    if 'dates' in data and data['dates']:
        for game in data['dates'][0]['games']:
            for side in ['away', 'home']:
                p = game['teams'][side].get('probablePitcher')
                if p:
                    starters.append({"Pitcher": p['fullName'].strip(), "Pitcher_ID": str(p['id']),
                                     "Team": game['teams'][side]['team']['name'],
                                     "Opponent": game['teams']['home' if side == 'away' else 'away']['team']['name']})
    return pd.DataFrame(starters)


@st.cache_data(ttl=3600)
def get_pitcher_baseline_data(pitcher_ids):
    params = {"personIds": ",".join(pitcher_ids),
              "hydrate": f"stats(group=[pitching],type=[season],season={datetime.now().year},gameType=R)"}
    data = fetch_json("https://statsapi.mlb.com/api/v1/people", params=params)
    stats_dict = {}
    if data and 'people' in data:
        for p in data['people']:
            p_id = str(p['id'])
            if 'stats' in p and p['stats']:
                s = p['stats'][0]['splits'][0]['stat']
                ip_val = safe_float(str(s.get('inningsPitched', "0")).replace('.1', '.333').replace('.2', '.666'))
                so, bb, hr, g = float(s.get('strikeOuts', 0)), float(s.get('baseOnBalls', 0)), float(
                    s.get('homeRuns', 0)), max(int(s.get('gamesPlayed', 1)), 1)
                stats_dict[p_id] = {"ERA": safe_float(s.get('era')), "WHIP": safe_float(s.get('whip')),
                                    "K%": round((so / max(int(s.get('battersFaced', 1)), 1)) * 100, 1),
                                    "IP/G": round(ip_val / g, 1),
                                    "Manual_FIP": round(((13 * hr) + (3 * bb) - (2 * so)) / max(ip_val, 1) + 3.2, 2)}
    return stats_dict


def build_platoon_row(split_name, stat_obj, sc_metrics):
    bf = stat_obj.get('battersFaced', 1)
    k = stat_obj.get('strikeOuts', 0)
    hits = stat_obj.get('hits', 0)
    doubles = stat_obj.get('doubles', 0)
    triples = stat_obj.get('triples', 0)
    hrs = stat_obj.get('homeRuns', 0)
    singles = hits - doubles - triples - hrs

    return {
        'SPLIT': split_name,
        'BF': bf,
        'HR': hrs,
        '1B': singles,
        '2B': doubles,
        '3B': triples,
        'BB': stat_obj.get('baseOnBalls', 0),
        'OBA': format_rate(stat_obj.get('obp', '.000')),
        'SLG': format_rate(stat_obj.get('slg', '.000')),
        'ISO': format_rate(safe_float(stat_obj.get('slg', 0)) - safe_float(stat_obj.get('avg', 0))),
        'BRL%': sc_metrics.get('BRL%', '.0%'),
        'HH%': sc_metrics.get('HH%', '.0%'),
        'K%': f"{round((k / max(bf, 1)) * 100, 1)}%"
    }


def build_standard_row(col_name, split_name, stat_obj):
    return {
        col_name: split_name,
        'AB': stat_obj.get('atBats', 0),
        'R': stat_obj.get('runs', 0),
        'H': stat_obj.get('hits', 0),
        '2B': stat_obj.get('doubles', 0),
        '3B': stat_obj.get('triples', 0),
        'HR': stat_obj.get('homeRuns', 0),
        'RBI': stat_obj.get('rbi', 0),
        'BB': stat_obj.get('baseOnBalls', 0),
        'HBP': stat_obj.get('hitBatsmen', 0),
        'SO': stat_obj.get('strikeOuts', 0),
        'SB': stat_obj.get('stolenBases', 0),
        'CS': stat_obj.get('caughtStealing', 0),
        'AVG': format_rate(stat_obj.get('avg', '.000')),
        'OBP': format_rate(stat_obj.get('obp', '.000')),
        'SLG': format_rate(stat_obj.get('slg', '.000')),
        'OPS': format_rate(stat_obj.get('ops', '.000'))
    }


def combine_stats(stat1, stat2):
    if not stat1: return stat2
    if not stat2: return stat1
    c = {}
    for key in ['atBats', 'runs', 'hits', 'doubles', 'triples', 'homeRuns', 'rbi', 'baseOnBalls', 'hitBatsmen',
                'strikeOuts', 'stolenBases', 'caughtStealing']:
        c[key] = stat1.get(key, 0) + stat2.get(key, 0)
    ab = c.get('atBats', 0)
    hits = c.get('hits', 0)
    bb = c.get('baseOnBalls', 0)
    hbp = c.get('hitBatsmen', 0)
    sf = stat1.get('sacFlies', 0) + stat2.get('sacFlies', 0)

    c['avg'] = hits / ab if ab > 0 else 0
    obp_denom = ab + bb + hbp + sf
    c['obp'] = (hits + bb + hbp) / obp_denom if obp_denom > 0 else 0
    tb = (hits - c['doubles'] - c['triples'] - c['homeRuns']) + (2 * c['doubles']) + (3 * c['triples']) + (
                4 * c['homeRuns'])
    c['slg'] = tb / ab if ab > 0 else 0
    c['ops'] = c['obp'] + c['slg']
    return c


# --- 3. DEEP DIVE DATA FETCHING ---
@st.cache_data(ttl=3600)
def process_pitcher_dashboard_data(first_name, last_name, year, pitcher_id):
    if not pitcher_id:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    top_table = []

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=[pitching],type=[season],season={year})"
        season_data = fetch_json(url)
        if season_data and 'people' in season_data and len(season_data['people']) > 0:
            stats_list = season_data['people'][0].get('stats', [])
            if stats_list and len(stats_list[0].get('splits', [])) > 0:
                api_stats = stats_list[0]['splits'][0]['stat']

                slg_val = safe_float(api_stats.get('slg', '.000'))
                avg_val = safe_float(api_stats.get('avg', '.000'))
                bf = max(int(api_stats.get('battersFaced', 1)), 1)
                k = int(api_stats.get('strikeOuts', 0))

                top_table.append({
                    'SPLIT': f"'{str(year)[2:]}",
                    'STARTS': api_stats.get('gamesStarted', 0),
                    'IP': api_stats.get('inningsPitched', '0.0'),
                    'ERA': api_stats.get('era', '0.00'),
                    'WHIP': api_stats.get('whip', '0.00'),
                    'OBA': format_rate(avg_val),
                    'ISO': format_rate(slg_val - avg_val),
                    'K%': f"{round((k / bf) * 100, 1)}%",
                    'K/9': api_stats.get('strikeoutsPer9Inn', '0.00'),
                    'HR/9': api_stats.get('homeRunsPer9', '0.00'),
                    'BRL%': '0.0%'
                })
    except Exception as e:
        pass

    sc_data = pd.DataFrame()
    splits_list = []
    tto_list = []
    order_list = []

    try:
        sc_data = statcast_pitcher(f'{year}-03-01', f'{year}-11-01', pitcher_id)
        if not sc_data.empty:
            overall_bbe = sc_data[sc_data['type'] == 'X']
            if len(overall_bbe) > 0:
                overall_brl = len(overall_bbe[overall_bbe['launch_speed_angle'] == 6])
                overall_brl_pct = (overall_brl / len(overall_bbe)) * 100
                if top_table:
                    top_table[0]['BRL%'] = f"{round(overall_brl_pct, 1)}%"

            sc_data = sc_data.sort_values(by=['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
            pas = sc_data[['game_pk', 'at_bat_number']].drop_duplicates(keep='first')
            pas['pa_rank'] = pas.groupby('game_pk').cumcount() + 1

            pas['TTTO'] = pas['pa_rank'].apply(
                lambda x: '1st TTO (1-9)' if x <= 9 else ('2nd TTO (10-18)' if x <= 18 else '3rd+ TTO (19+)'))
            pas['Order_Slot'] = ((pas['pa_rank'] - 1) % 9) + 1
            pas['Order_Slot_Label'] = pas['Order_Slot'].apply(lambda x: f"Batting #{x}")

            sc_data = sc_data.merge(pas[['game_pk', 'at_bat_number', 'TTTO', 'Order_Slot_Label']],
                                    on=['game_pk', 'at_bat_number'], how='left')

            def get_split_stats(subset_pitches):
                # FIX: Keep the LAST pitch of the at-bat to capture the terminal event
                terminal_events = subset_pitches.dropna(subset=['events']).drop_duplicates(
                    subset=['game_pk', 'at_bat_number'], keep='last')
                bf = len(terminal_events)
                if bf == 0: return None

                ev_counts = terminal_events['events'].value_counts()
                hr = ev_counts.get('home_run', 0)
                single = ev_counts.get('single', 0)
                double = ev_counts.get('double', 0)
                triple = ev_counts.get('triple', 0)
                bb = ev_counts.get('walk', 0) + ev_counts.get('intent_walk', 0)
                k = ev_counts.get('strikeout', 0) + ev_counts.get('strikeout_double_play', 0)
                hbp = ev_counts.get('hit_by_pitch', 0)
                sf = ev_counts.get('sac_fly', 0) + ev_counts.get('sac_fly_double_play', 0)
                sac = ev_counts.get('sac_bunt', 0) + ev_counts.get('sac_bunt_double_play', 0)
                ci = ev_counts.get('catcher_interf', 0)

                hits = single + double + triple + hr
                ab = bf - bb - hbp - sf - sac - ci

                oba = (hits + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) > 0 else 0
                slg = (single + 2 * double + 3 * triple + 4 * hr) / ab if ab > 0 else 0
                iso = slg - (hits / ab if ab > 0 else 0)
                k_pct = (k / bf * 100) if bf > 0 else 0
                avg = hits / ab if ab > 0 else 0
                ops = oba + slg

                bbe = subset_pitches[subset_pitches['type'] == 'X']
                barrels = bbe[bbe['launch_speed_angle'] == 6]
                hard_hit = bbe[bbe['launch_speed'] >= 95]
                brl_pct = (len(barrels) / len(bbe) * 100) if len(bbe) > 0 else 0
                hh_pct = (len(hard_hit) / len(bbe) * 100) if len(bbe) > 0 else 0

                return {
                    'BF': bf, 'AB': ab, 'H': hits, '1B': single, '2B': double, '3B': triple, 'HR': hr,
                    'BB': bb, 'HBP': hbp, 'SO': k, 'AVG': avg, 'OBA': oba, 'OBP': oba, 'SLG': slg,
                    'OPS': ops, 'ISO': iso, 'K%': k_pct, 'BRL%': brl_pct, 'HH%': hh_pct
                }

            for name, stand in [('vs RHB', 'R'), ('vs LHB', 'L')]:
                stats = get_split_stats(sc_data[sc_data['stand'] == stand])
                if stats:
                    splits_list.append({
                        'SPLIT': f"'{str(year)[2:]} {name}",
                        'BF': stats['BF'], 'HR': stats['HR'], '1B': stats['1B'], '2B': stats['2B'], '3B': stats['3B'],
                        'BB': stats['BB'], 'OBA': format_rate(stats['OBA']), 'SLG': format_rate(stats['SLG']),
                        'ISO': format_rate(stats['ISO']), 'BRL%': f"{round(stats['BRL%'], 1)}%",
                        'HH%': f"{round(stats['HH%'], 1)}%", 'K%': f"{round(stats['K%'], 1)}%"
                    })

            for name in ['1st TTO (1-9)', '2nd TTO (10-18)', '3rd+ TTO (19+)']:
                stats = get_split_stats(sc_data[sc_data['TTTO'] == name])
                if stats:
                    tto_list.append({
                        'TTO SPLIT': name,
                        'AB': stats['AB'], 'H': stats['H'], '2B': stats['2B'], '3B': stats['3B'], 'HR': stats['HR'],
                        'BB': stats['BB'], 'HBP': stats['HBP'], 'SO': stats['SO'], 'AVG': format_rate(stats['AVG']),
                        'OBP': format_rate(stats['OBP']), 'SLG': format_rate(stats['SLG']),
                        'OPS': format_rate(stats['OPS'])
                    })

            for i in range(1, 10):
                name = f"Batting #{i}"
                stats = get_split_stats(sc_data[sc_data['Order_Slot_Label'] == name])
                if stats:
                    order_list.append({
                        'BATTING ORDER': name,
                        'AB': stats['AB'], 'H': stats['H'], '2B': stats['2B'], '3B': stats['3B'], 'HR': stats['HR'],
                        'BB': stats['BB'], 'HBP': stats['HBP'], 'SO': stats['SO'], 'AVG': format_rate(stats['AVG']),
                        'OBP': format_rate(stats['OBP']), 'SLG': format_rate(stats['SLG']),
                        'OPS': format_rate(stats['OPS'])
                    })

    except Exception as e:
        st.error(f"Statcast calculation error: {e}")

    top_df = pd.DataFrame(top_table)
    platoon_df = pd.DataFrame(splits_list)
    tto_df = pd.DataFrame(tto_list)
    order_df = pd.DataFrame(order_list)

    return top_df, platoon_df, tto_df, order_df, sc_data


# --- 4. MAIN DASHBOARD EXECUTION ---
st.title("🛡️ Pitching Engine (V6.1 Predictive)")
probables = get_live_probables()

if not probables.empty:
    with st.spinner("Compiling Telemetry..."):
        baseline = get_pitcher_baseline_data(probables['Pitcher_ID'].tolist())
        statcast_map = get_league_statcast_map()
        results = []
        for _, row in probables.iterrows():
            p_id = row['Pitcher_ID']
            s = baseline.get(p_id, {"ERA": 0.0, "K%": 0.0, "WHIP": 0.0, "IP/G": 0.0, "Manual_FIP": 0.0})
            sc = statcast_map.get(p_id, {"xERA": s.get('ERA', 0), "BRL%": 0.0, "FIP": s.get('Manual_FIP', 0)})
            active_fip = sc.get('FIP') if sc.get('FIP', 0) > 0 else s.get('Manual_FIP', 0)
            reg_idx = round(s.get('ERA', 0) - active_fip, 2)
            results.append({**row, **s, **sc, "Reg_Idx": reg_idx,
                            "SO Score": round((s.get('K%', 0) * 2.5) + (s.get('IP/G', 0) * 4.0), 1),
                            "Outs Score": round((s.get('IP/G', 0) * 20.0) - (s.get('WHIP', 0) * 10.0), 1),
                            "Fade Score": round((reg_idx * 20.0) + (sc.get('BRL%', 0) * 3.0), 1),
                            "Proj SO": round((s.get('K%', 0) / 100) * 28, 1), "Proj Outs": round(
                    min(20.0, (s.get('IP/G', 0) * 3) * (1 / (1 + (s.get('WHIP', 1.25) - 1.25) * 0.2))), 1)})

        df = pd.DataFrame(results)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🔥 Strikeout Targets")
        st.dataframe(
            df.nlargest(5, "SO Score")[['Pitcher', 'Opponent', 'Proj SO', 'SO Score']], hide_index=True)
    with c2:
        st.subheader("🛡️ Outs Recorded")
        st.dataframe(
            df.nlargest(5, "Outs Score")[['Pitcher', 'Opponent', 'Proj Outs', 'Outs Score']], hide_index=True)
    with c3:
        st.subheader("🚨 Fade Targets (Regress)")
        st.dataframe(
            df.nlargest(5, "Fade Score")[['Pitcher', 'Opponent', 'Reg_Idx', 'Fade Score']], hide_index=True)

    with st.expander("📋 Master Discord Export"):
        export_text = "**🎯 H2 SPORTS MASTER PITCHING PROPS**\n\n🔥 **Strikeout Ticket:**\n" + "\n".join(
            [f"{i + 1}. {r['Pitcher']} (Proj: {r['Proj SO']})" for i, r in enumerate(
                df.nlargest(5, "SO Score").to_dict('records'))]) + "\n\n🛡️ **Efficiency Ticket (Outs):**\n" + "\n".join(
            [f"{i + 1}. {r['Pitcher']} (Proj: {r['Proj Outs']})" for i, r in enumerate(
                df.nlargest(5, "Outs Score").to_dict(
                    'records'))]) + "\n\n🚨 **Fade Targets (High Regress):**\n" + "\n".join(
            [f"{i + 1}. {r['Pitcher']} (vs {r['Opponent']})" for i, r in
             enumerate(df.nlargest(5, "Fade Score").to_dict('records'))])

        st.code(export_text, language="markdown")

    st.dataframe(df.drop(columns=['SO Score', 'Outs Score', 'Fade Score']), use_container_width=True, hide_index=True)

    # --- 5. PITCHER DEEP DIVE ---
    st.markdown("---")
    st.subheader("⚾ Pitcher Deep Dive: Split Matrix & Pitch Mix Visualizer")

    selected_p = st.selectbox("Select Active Starting Pitcher to Inspect:", df['Pitcher'].tolist())

    if selected_p:
        p_id = int(df[df['Pitcher'] == selected_p]['Pitcher_ID'].values[0])
        clean_name = strip_accents(selected_p)
        name_parts = clean_name.split(' ', 1)
        first = name_parts[0]
        last = name_parts[1] if len(name_parts) > 1 else ""

        with st.spinner(f"Extracting Live APIs for {selected_p}..."):
            season_df, platoon_df, tto_df, order_df, raw_sc = process_pitcher_dashboard_data(first, last,
                                                                                             datetime.now().year, p_id)

            col1, col2 = st.columns([7, 3])

            with col1:
                st.markdown("#### Baseline Cumulative Season Layout")
                if season_df is not None and not season_df.empty:
                    st.dataframe(season_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Season summary stats missing for selection.")

                st.markdown("#### Handedness Split Matrix")
                if platoon_df is not None and not platoon_df.empty:
                    st.dataframe(platoon_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No split profiles compiled yet for this matchup window.")

                st.markdown("#### Times Through Order (Slash Line Matrix)")
                if tto_df is not None and not tto_df.empty:
                    st.dataframe(tto_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No TTO profiles compiled yet for this matchup window.")

                st.markdown("#### Batting Order Vulnerability (Slots 1-9)")
                if order_df is not None and not order_df.empty:
                    st.dataframe(order_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No Batting Order profiles compiled yet.")

            with col2:
                st.markdown("#### Pitch-Mix Distribution & Frequency")
                if raw_sc is not None and not raw_sc.empty:
                    pitch_counts = raw_sc['pitch_type'].value_counts()

                    if not pitch_counts.empty:
                        chart_c1, chart_c2 = st.columns(2)

                        with chart_c1:
                            st.bar_chart(pitch_counts, height=180)

                        with chart_c2:
                            fig, ax = plt.subplots(figsize=(2.5, 2.5))
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                            ax.pie(pitch_counts, labels=pitch_counts.index, autopct='%1.1f%%', startangle=140,
                                   colors=colors[:len(pitch_counts)], textprops={'fontsize': 6})
                            ax.axis('equal')
                            fig.patch.set_facecolor('#0e1117')
                            plt.title("Arsenal Mix", color="white", fontsize=9)
                            for text in ax.texts:
                                text.set_color('white')

                            st.pyplot(fig, use_container_width=True)
                    else:
                        st.write("Pitch inventory tracking unavailable.")
                else:
                    st.warning("Awaiting raw pitch sequence collection mapping.")

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("#### Vulnerability Hotspots (Allowed SLG)")
                if order_df is not None and not order_df.empty:
                    # Figsize reduced from (5,3.5) to (4, 2.5)
                    fig_vuln, ax_vuln = plt.subplots(figsize=(4, 2.5))
                    plot_df = order_df.copy()

                    plot_df['SLG_num'] = plot_df['SLG'].apply(lambda x: float(f"0{x}" if str(x).startswith('.') else x))

                    bars = ax_vuln.bar(plot_df['BATTING ORDER'].str.replace('Batting #', 'Slot '), plot_df['SLG_num'],
                                       color='#ff4b4b', edgecolor='white', width=0.6)

                    fig_vuln.patch.set_facecolor('#0e1117')
                    ax_vuln.set_facecolor('#0e1117')
                    ax_vuln.spines['bottom'].set_color('white')
                    ax_vuln.spines['left'].set_color('white')
                    ax_vuln.spines['top'].set_visible(False)
                    ax_vuln.spines['right'].set_visible(False)

                    ax_vuln.tick_params(colors='white', labelsize=7)
                    ax_vuln.set_ylabel("Allowed SLG", color='white', fontsize=8)
                    ax_vuln.set_xlabel("Batting Order Slot", color='white', fontsize=8)

                    for bar in bars:
                        height = bar.get_height()
                        ax_vuln.annotate(f'.{int(height * 1000):03d}' if height < 1 else f'{height:.3f}',
                                         xy=(bar.get_x() + bar.get_width() / 2, height),
                                         xytext=(0, 2),
                                         textcoords="offset points",
                                         ha='center', va='bottom', color='white', fontsize=7, weight='bold')

                    st.pyplot(fig_vuln, use_container_width=True)
                else:
                    st.info("No Batting Order profiles compiled yet.")

else:
    st.info("Waiting for API data...")