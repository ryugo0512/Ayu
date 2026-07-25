import datetime
import json
import os
import re
import time
import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="北海道 鮎コンディション判定", page_icon="🐟", layout="wide")

LOG_FILE = "fishing_logs.json"
WATER_TEMP_LOG_FILE = "water_temp_logs.json"
WATER_LOG_FILE = "water_levels_history.json"

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_logs(logs):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def save_log(log_entry):
    logs = load_logs()
    logs.append(log_entry)
    save_logs(logs)

def delete_log(index):
    logs = load_logs()
    if 0 <= index < len(logs):
        logs.pop(index)
        save_logs(logs)

def load_water_temp_logs():
    if os.path.exists(WATER_TEMP_LOG_FILE):
        with open(WATER_TEMP_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_water_temp_logs(logs):
    with open(WATER_TEMP_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def save_water_temp_log(log_entry):
    logs = load_water_temp_logs()
    logs.append(log_entry)
    save_water_temp_logs(logs)

def delete_water_temp_log(index):
    logs = load_water_temp_logs()
    if 0 <= index < len(logs):
        logs.pop(index)
        save_water_temp_logs(logs)

def load_water_history():
    if os.path.exists(WATER_LOG_FILE):
        with open(WATER_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_water_history(river_name, timestamp_str, level):
    history = load_water_history()
    if river_name not in history:
        history[river_name] = {}
    history[river_name][timestamp_str] = level
    with open(WATER_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def estimate_dynamic_decay_rate(river_name, base_level, default_decay):
    history = load_water_history().get(river_name, {})
    if len(history) < 10:
        return default_decay
    sorted_times = sorted(history.keys())
    ratios = []
    for i in range(1, len(sorted_times)):
        t_prev, t_curr = sorted_times[i - 1], sorted_times[i]
        l_prev, l_curr = history[t_prev], history[t_curr]
        try:
            dt = (pd.to_datetime(t_curr) - pd.to_datetime(t_prev)).total_seconds() / 3600.0
            if 0.8 <= dt <= 1.5:
                if l_prev > base_level and l_curr > base_level and l_curr < l_prev:
                    ratio = ((l_curr - base_level) / (l_prev - base_level)) ** (1.0 / dt)
                    if 0.90 <= ratio <= 0.9999:
                        ratios.append(ratio)
        except Exception:
            continue
    if len(ratios) >= 5:
        return float(np.median(ratios))
    return default_decay

def estimate_water_temp_bias(river_name, river_info):
    temp_logs = load_water_temp_logs()
    river_logs = [l for l in temp_logs if l.get("river") == river_name and "measured_water_temp" in l]
    if not river_logs:
        return 0.0
    biases = []
    for l in river_logs:
        try:
            l_date = datetime.date.fromisoformat(l["date"])
            day_of_year = l_date.timetuple().tm_yday
            calc_base = river_info["temp_base"] + 2.0 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
            biases.append(l["measured_water_temp"] - (calc_base + (20.0 * river_info["temp_factor"])))
        except Exception:
            continue
    if biases:
        return float(np.median(biases))
    return 0.0

RIVERS = {
    "尻別川本流（豊国橋）": {
        "lat": 42.8021, "lon": 140.5251, "base_level": 9.08, "default_actual": 9.08,
        "station_name": "豊国橋", "river_system": "尻別川水系 尻別川",
        "weather_url": "https://weathernews.jp/onebox/river/shiribetsugawa/?pid=2078700400004",
        "temp_base": 9.0, "temp_factor": 0.30, "max_temp": 20.0, "decay_rate": 0.9975,
    },
    "昆布川（昆布）": {
        "lat": 42.7958, "lon": 140.5986, "base_level": 43.58, "default_actual": 43.58,
        "station_name": "昆布川橋", "river_system": "尻別川水系 昆布川",
        "weather_url": "https://weathernews.jp/onebox/river/shiribetsugawa/?pid=0025700400389",
        "temp_base": 8.5, "temp_factor": 0.32, "max_temp": 19.5, "decay_rate": 0.9970,
    },
    "天ノ川（上ノ国）": {
        "lat": 41.7997, "lon": 140.1163, "base_level": 1.60, "default_actual": 1.60,
        "station_name": "古守大橋", "river_system": "天ノ川水系 天ノ川",
        "weather_url": "https://weathernews.jp/onebox/river/?pid=0025700400132",
        "temp_base": 10.0, "temp_factor": 0.35, "max_temp": 21.0, "decay_rate": 0.9975,
    },
    "朱太川（黒松内）": {
        "lat": 42.6683, "lon": 140.3061, "base_level": 1.44, "default_actual": 1.44,
        "station_name": "朱太川実橋", "river_system": "朱太川水系 朱太川",
        "weather_url": "https://weathernews.jp/onebox/river/shubutogawa/?pid=0025700400387",
        "temp_base": 9.5, "temp_factor": 0.32, "max_temp": 20.5, "decay_rate": 0.9972,
    },
}

@st.cache_data(ttl=600)
def fetch_weather_water_level(url, default_val):
    if not url: return default_val, "デフォルト値"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(2):
        try:
            res = requests.get(url, headers=headers, timeout=3)
            res.raise_for_status()
            clean_text = " ".join(re.sub(r"<[^>]+>", " ", res.text).split())
            match = re.search(r"現在水位\s*(\d+\.\d{2})\s*m", clean_text)
            if match: return float(match.group(1)), "自動取得"
            match = re.search(r"\d{1,2}:\d{2}\s*時点\s*(\d+\.\d{2})\s*m", clean_text)
            if match: return float(match.group(1)), "自動取得"
            match = re.search(r"時点\s*(\d+\.\d{2})\s*m", clean_text)
            if match: return float(match.group(1)), "自動取得"
            matches = re.findall(r"(\d+\.\d{2})\s*m", clean_text)
            if matches:
                for m_str in matches:
                    val = float(m_str)
                    if 0.001 < abs(val - default_val) <= 3.0: return val, "自動取得"
                return float(matches[0]), "自動取得"
            return default_val, "デフォルト(未検出)"
        except Exception:
            if attempt == 0: time.sleep(2)
    return default_val, "デフォルト(通信エラー)"
@st.cache_data(ttl=3600)
def fetch_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,weathercode,sunshine_duration,shortwave_radiation,windspeed_10m&windspeed_unit=ms&past_days=7&forecast_days=16&timezone=Asia%2FTokyo"
    for attempt in range(2):
        try:
            res = requests.get(url, timeout=5)
            res.raise_for_status()
            data = res.json()
            if "hourly" in data:
                df = pd.DataFrame(data["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                return df, True
        except Exception:
            if attempt == 0: time.sleep(2)
    now = pd.Timestamp.now().floor("h")
    df_dummy = pd.DataFrame({
        "time": pd.date_range(end=now + pd.Timedelta(days=16), periods=24 * 16, freq="h"),
        "temperature_2m": 20.0, "precipitation": 0.0, "weathercode": 0,
        "shortwave_radiation": 200.0, "windspeed_10m": 2.0,
    })
    return df_dummy, False

def get_weather_desc(code):
    if code in [0]: return "快晴"
    elif code in [1, 2]: return "晴れ時々曇り"
    elif code in [3]: return "曇り"
    elif code in [45, 48]: return "霧"
    elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return "雨"
    elif code in [95, 96, 99]: return "雷雨"
    return "曇り"

def simulate_water_levels(df_weather, base_level, current_actual, river_decay_rate, river_name):
    if df_weather is None or df_weather.empty or "precipitation" not in df_weather.columns:
        return pd.DataFrame({"time": [pd.Timestamp.now()], "simulated_level": [current_actual]})
    df_weather = df_weather.sort_values("time").reset_index(drop=True)
    history = load_water_history().get(river_name, {})
    now = pd.Timestamp.now().floor("h")
    time_diffs = (df_weather["time"] - now).abs()
    now_idx = int(time_diffs.idxmin())
    simulated_levels = np.full(len(df_weather), np.nan)
    simulated_levels[now_idx] = current_actual
    for i in range(now_idx - 1, -1, -1):
        t_str = df_weather.loc[i, "time"].strftime("%Y-%m-%d %H:00")
        simulated_levels[i] = history.get(t_str, np.nan)
    curr_lvl = current_actual
    eff_decay = np.exp(-np.log(2) / 48.0)
    fut_eff_rain = 0.0
    for i in range(now_idx + 1, len(df_weather)):
        rain = df_weather.loc[i, "precipitation"]
        fut_eff_rain = fut_eff_rain * eff_decay + rain
        rain_imp = rain * 0.035 + fut_eff_rain * 0.001
        diff_from_base = curr_lvl - base_level
        if diff_from_base > 0:
            next_diff = diff_from_base * river_decay_rate + rain_imp
        else:
            next_diff = diff_from_base + rain_imp
        curr_lvl = base_level + max(-0.25, next_diff)
        simulated_levels[i] = curr_lvl
    df_weather["simulated_level"] = simulated_levels
    return df_weather

def analyze_condition(df_weather, is_weather_live, river_info, user_logs, target_river, target_date, current_actual):
    effective_base = river_info["base_level"]
    river_decay_rate = estimate_dynamic_decay_rate(target_river, effective_base, river_info.get("decay_rate", 0.9975))
    temp_bias = estimate_water_temp_bias(target_river, river_info)
    df_weather = simulate_water_levels(df_weather, effective_base, current_actual, river_decay_rate, target_river)
    target_datetime = datetime.datetime.combine(target_date, datetime.time(12, 0))
    bias_growth = np.mean([l.get("moss_feedback", 0) for l in user_logs if l.get("river") == target_river] or [0]) * 0.1
    day_of_year = target_date.timetuple().tm_yday
    adjusted_temp_base = river_info["temp_base"] + 2.0 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
    temp_col = "temperature_2m" if "temperature_2m" in df_weather.columns else df_weather.columns[1]
    df_weather["estimated_water_temp"] = np.minimum(adjusted_temp_base + (df_weather[temp_col] * river_info["temp_factor"]) + temp_bias, river_info["max_temp"])
    target_df = df_weather[df_weather["time"].dt.date == target_date].copy() if "time" in df_weather.columns else pd.DataFrame()
    df_past = df_weather[df_weather["time"] <= target_datetime].copy() if "time" in df_weather.columns else df_weather.copy()
    if "precipitation" in df_past.columns:
        df_past["rain_12h"] = df_past["precipitation"].rolling(12, min_periods=1).sum()
        heavy_events = df_past[(df_past["precipitation"] >= 30.0) | (df_past["rain_12h"] >= 60.0)]
        days_since_flood = (target_datetime - heavy_events["time"].max()).days if not heavy_events.empty else 10
    else:
        days_since_flood = 10
    recent_rain = df_past.tail(24)["precipitation"].sum() if "precipitation" in df_past.columns else 0.0
    clarity_recovery = "強濁り" if recent_rain > 60 else "笹濁り" if recent_rain > 30 else "清澄"
    clarity_score = 1 if recent_rain > 60 else 2 if recent_rain > 30 else 3
    m, d = target_date.month, target_date.day
    if m == 7 and d <= 15: season_mode, growth_rate = "初期", 9.0
    elif (m == 7 and d > 15) or (m == 8 and d <= 15): season_mode, growth_rate = "盛期", 12.5
    elif m == 8 and d > 15: season_mode, growth_rate = "晩夏", 10.0
    else: season_mode, growth_rate = "終盤", 7.0
    recent_rad = df_past.tail(max(24, days_since_flood * 24))["shortwave_radiation"].mean() if "shortwave_radiation" in df_past.columns else 150.0
    moss_growth = min(100, int((days_since_flood * growth_rate * max(0.7, min(1.3, recent_rad / 180.0))) * (1.0 + bias_growth)))
    if not target_df.empty and len(target_df) >= 24:
        hourly_water_temp = target_df["estimated_water_temp"].tolist()[:24]
        display_water_level = current_actual if target_date == datetime.date.today() else target_df["simulated_level"].mean()
        weather_desc = get_weather_desc(target_df["weathercode"].mode()[0] if "weathercode" in target_df.columns else 0)
        temp_max, temp_min = target_df["temperature_2m"].max(), target_df["temperature_2m"].min()
        water_temp_max, water_temp_avg = max(hourly_water_temp), float(np.mean(hourly_water_temp))
        max_wind = target_df["windspeed_10m"].max()
    else:
        hourly_water_temp = [14.0 + (i if i <= 14 else 28 - i) * 0.3 for i in range(24)]
        display_water_level = current_actual
        weather_desc, temp_max, temp_min, water_temp_max, water_temp_avg, max_wind = "晴れ", 22.0, 16.0, 17.5, 15.8, 2.0
    level_diff = display_water_level - effective_base
    if level_diff < -0.10: level_trend = f"渇水 ({level_diff*100:+.0f}cm)"
    elif level_diff <= 0.15: level_trend = f"平水 ({level_diff*100:+.0f}cm)"
    elif level_diff <= 0.40: level_trend = f"やや高水 ({level_diff*100:+.0f}cm)"
    else: level_trend = f"大増水 ({level_diff*100:+.0f}cm)"
    if days_since_flood <= 1 or moss_growth < 20: moss_alert = "全飛び直後"
    elif days_since_flood <= 3 or moss_growth < 50: moss_alert = "垢付き始め"
    elif level_diff < -0.15 and days_since_flood > 10: moss_alert = "垢腐り注意"
    else: moss_alert = "新垢良好"
    df_future = df_weather[df_weather["time"] >= target_datetime].head(24) if "time" in df_weather.columns else pd.DataFrame()
    fut_rain = df_future["precipitation"].sum() if "precipitation" in df_future.columns else 0.0
    flood_risk = "警戒" if fut_rain > 50.0 else "注意" if fut_rain > 25.0 else "安定"
    temp_pts = 3 if len([t for t in hourly_water_temp if t >= 18.0]) >= 4 else (2 if len([t for t in hourly_water_temp if t >= 18.0]) >= 2 else 1)
    raw_score = int((moss_growth / 100) * 4) + clarity_score + temp_pts
    score = max(1, min(raw_score, 3 if days_since_flood <= 2 else 5 if days_since_flood <= 4 else 10))
    if level_diff <= -0.30: score -= 3
    elif level_diff <= -0.20: score -= 2
    elif level_diff < 0: score -= 1
    score = max(1, min(score, 1 if level_diff >= 0.50 else 3 if level_diff >= 0.30 else 7 if level_diff >= 0.15 else 10))
    df_hydro = df_weather.copy()
    df_hydro["base_level"] = effective_base
    return {
        "water_level": display_water_level, "level_trend": level_trend, "days_since_flood": days_since_flood,
        "moss_growth": moss_growth, "moss_alert": moss_alert, "flood_risk": flood_risk, "clarity_recovery": clarity_recovery,
        "season_mode": season_mode, "score": score, "hourly_water_temp": hourly_water_temp, "df_hydro": df_hydro,
        "target_df": target_df, "weather_desc": weather_desc, "temp_max": temp_max, "temp_min": temp_min,
        "water_temp_max": water_temp_max, "water_temp_avg": water_temp_avg, "max_wind": max_wind, "level_diff": level_diff,
        "has_precipitation_data": ("precipitation" in df_weather.columns), "learned_decay": river_decay_rate, "learned_temp_bias": temp_bias
    }
st.title("北海道 鮎コンディション判定")

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    target_river = st.selectbox("河川を選択", list(RIVERS.keys()))
with col_sel2:
    today_date = datetime.date.today()
    target_date = st.date_input("釣行予定日", today_date, min_value=today_date - datetime.timedelta(days=7), max_value=today_date + datetime.timedelta(days=5))

river_info = RIVERS[target_river]

current_actual, fetch_source = fetch_weather_water_level(river_info["weather_url"], river_info["default_actual"])
now_hour_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:00")
save_water_history(target_river, now_hour_str, current_actual)

df_weather, is_weather_live = fetch_weather_data(river_info["lat"], river_info["lon"])
user_logs = load_logs()
res = analyze_condition(df_weather, is_weather_live, river_info, user_logs, target_river, target_date, current_actual)

st.caption(f"観測所: {river_info['station_name']} / 基準水位設定: {river_info['base_level']:.2f}m / 現在実測値: {current_actual:.2f}m ({fetch_source})")

st.markdown("---")
st.subheader("コンディション予測")
st.markdown(f"釣行日おすすめ度 : {res['score']} / 10")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("水位状況", f"{res['water_level']:.2f} m", res["level_trend"])
col2.metric("天気", res["weather_desc"])
col3.metric("気温", f"{res['temp_max']:.1f}℃", f"最低 {res['temp_min']:.1f}℃")
col4.metric("水温", f"{res['water_temp_max']:.1f}℃", f"平均 {res['water_temp_avg']:.1f}℃")
col5.metric("ハミ垢", f"{res['moss_growth']} %")
col6.metric("風速", f"{res['max_wind']:.1f} m/s")

st.write(f"大水からの経過日数: {res['days_since_flood']} 日 / 濁り予測: {res['clarity_recovery']} / アラート: {res['moss_alert']}")

st.markdown("---")
st.subheader("水位グラフ")
graph_range = st.radio("グラフ表示期間", ["直近2日間", "直近1週間"], horizontal=True)
if not res["df_hydro"].empty and "time" in res["df_hydro"].columns:
    past_days = 7 if graph_range == "直近1週間" else 2
    start_time = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=past_days))
    end_time = pd.to_datetime(target_date + datetime.timedelta(days=1))
    chart_hydro = res["df_hydro"][(res["df_hydro"]["time"] >= start_time) & (res["df_hydro"]["time"] < end_time)].copy()
    if not chart_hydro.empty:
        chart_hydro["水位(m)"] = chart_hydro["simulated_level"]
        chart_hydro["時間"] = chart_hydro["time"].dt.strftime("%m/%d %H時")
        chart_hydro = chart_hydro.rename(columns={"base_level": "基準水位線(m)"})
        min_val, max_val = chart_hydro[["水位(m)", "基準水位線(m)"]].min().min(), chart_hydro[["水位(m)", "基準水位線(m)"]].max().max()
        hydro_melt = chart_hydro.melt(id_vars=["時間"], value_vars=["水位(m)", "基準水位線(m)"], var_name="凡例", value_name="水位")
        hydro_chart = alt.Chart(hydro_melt).mark_line(strokeWidth=2).encode(
            x=alt.X("時間:N", sort=None), y=alt.Y("水位:Q", scale=alt.Scale(domain=[min_val - 0.1, max_val + 0.1])),
            color="凡例:N", tooltip=["時間", "凡例", "水位"]
        ).properties(height=300)
        st.altair_chart(hydro_chart, use_container_width=True)

st.markdown("---")
st.subheader("水温グラフ")
if not res["df_hydro"].empty and "time" in res["df_hydro"].columns and "estimated_water_temp" in res["df_hydro"].columns:
    past_days = 7 if graph_range == "直近1週間" else 2
    start_time = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=past_days))
    end_time = pd.to_datetime(target_date + datetime.timedelta(days=1))
    chart_temp = res["df_hydro"][(res["df_hydro"]["time"] >= start_time) & (res["df_hydro"]["time"] < end_time)].copy()
    if not chart_temp.empty:
        chart_temp["推定水温(℃)"] = chart_temp["estimated_water_temp"]
        chart_temp["時間"] = chart_temp["time"].dt.strftime("%m/%d %H時")
        temp_chart = alt.Chart(chart_temp).mark_line(strokeWidth=2, color="orange").encode(
            x=alt.X("時間:N", sort=None), y=alt.Y("推定水温(℃):Q", scale=alt.Scale(zero=False)),
            tooltip=["時間", "推定水温(℃)"]
        ).properties(height=250)
        st.altair_chart(temp_chart, use_container_width=True)

st.markdown("---")
st.subheader("各種ログ保存")
with st.form("water_temp_form"):
    c1, c2, c3 = st.columns(3)
    wt_date = c1.date_input("水温測定日", today_date)
    wt_val = c2.number_input("実測水温(℃)", value=16.0, step=0.1)
    wt_time = c3.time_input("測定時間")
    if st.form_submit_button("水温保存"):
        save_water_temp_log({"date": str(wt_date), "river": target_river, "measured_water_temp": wt_val, "water_temp_time": wt_time.strftime("%H:%M")})
        st.rerun()

with st.form("log_form"):
    c1, c2 = st.columns(2)
    log_date = c1.date_input("釣行日", today_date)
    catch_cnt = c2.number_input("釣果", value=10)
    moss = st.select_slider("垢状況", ["全飛直後", "薄っすら新垢", "ベスト", "垢腐り"], "ベスト")
    feedback = {"全飛直後": -2, "薄っすら新垢": -1, "ベスト": 0, "垢腐り": 1}
    if st.form_submit_button("釣果保存"):
        save_log({"date": str(log_date), "river": target_river, "catch": catch_cnt, "moss_condition": moss, "moss_feedback": feedback[moss]})
        st.rerun()
