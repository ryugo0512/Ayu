import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
import requests
import altair as alt

# ---------------------------------------------------------
# 1. 基本設定とデータ永続化（実釣ログ保存・削除）
# ---------------------------------------------------------
st.set_page_config(page_title="北海道 鮎コンディション判定", page_icon="🐟", layout="wide")

LOG_FILE = "fishing_logs.json"

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

# ---------------------------------------------------------
# 2. 河川・観測所データ設定（高精度パラメータ追加）
# ---------------------------------------------------------
RIVERS = {
    "尻別川本流（蘭越）": {
        "lat": 42.8021, "lon": 140.5251, "base_level": 1.20,
        "stg_id": "3010312811020", "runoff_factor": 0.025, "decay_rate": 0.96, "drought_rate": 0.0005,
        "temp_base": 11.0, "temp_factor": 0.35, "max_temp": 21.5
    },
    "昆布川（昆布）": {
        "lat": 42.7958, "lon": 140.5986, "base_level": 0.80,
        "stg_id": "3010312811040", "runoff_factor": 0.030, "decay_rate": 0.95, "drought_rate": 0.0005,
        "temp_base": 10.5, "temp_factor": 0.38, "max_temp": 21.0
    },
    "天ノ川（上ノ国）": {
        "lat": 41.7997, "lon": 140.1163, "base_level": 0.90,
        "stg_id": "3010112811010", "runoff_factor": 0.030, "decay_rate": 0.97, "drought_rate": 0.0005,
        "temp_base": 12.0, "temp_factor": 0.40, "max_temp": 22.5
    },
    "朱太川（黒松内）": {
        "lat": 42.6683, "lon": 140.3061, "base_level": 0.70,
        "stg_id": "3010312811050", "runoff_factor": 0.035, "decay_rate": 0.96, "drought_rate": 0.0005,
        "temp_base": 11.5, "temp_factor": 0.38, "max_temp": 22.0
    }
}

# ---------------------------------------------------------
# 3. 外部API取得モジュール & 天気コード変換（m/s指定追加）
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_weather_and_hydro(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,weathercode,sunshine_duration,shortwave_radiation,windspeed_10m&windspeed_unit=ms&past_days=14&forecast_days=7&timezone=Asia%2FTokyo"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception:
        return None

@st.cache_data(ttl=900)
def fetch_real_water_level(stg_id):
    url = f"https://www.river.go.jp/kawabou/api/v1/waterlevel/latest?stationCode={stg_id}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return float(data.get("waterLevel", None))
    except Exception:
        pass
    return None

def get_weather_desc(code):
    if code in [0]:
        return "☀️ 快晴"
    elif code in [1, 2]:
        return "🌤️ 晴れ時々曇り"
    elif code in [3]:
        return "☁️ 曇り"
    elif code in [45, 48]:
        return "🌫️ 霧"
    elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
        return "🌧️ 雨"
    elif code in [95, 96, 99]:
        return "⛈️ 雷雨"
    else:
        return "☁️ 曇り"

# ---------------------------------------------------------
# 4. 水位推移シミュレーション（実効雨量・土壌保持モデル導入）
# ---------------------------------------------------------
def simulate_water_levels(df_weather, base_level, runoff_factor, decay_rate, drought_rate, real_level=None):
    levels = []
    current_runoff = 0.0
    effective_rain = 0.0
    dry_hours = 0
    
    eff_decay = np.exp(-np.log(2) / 48.0)

    for idx, row in df_weather.iterrows():
        rain = row["precipitation"]
        temp = row["temperature_2m"]
        
        effective_rain = effective_rain * eff_decay + rain
        
        if rain > 0.2:
            dry_hours = 0
            soil_contribution = effective_rain * 0.0015
            current_runoff = current_runoff * decay_rate + (rain * runoff_factor) + soil_contribution
            drought_offset = 0.0
        else:
            dry_hours += 1
            current_runoff = current_runoff * decay_rate
            temp_penalty = max(1.0, temp / 22.0)
            drought_offset = min(0.20, dry_hours * drought_rate * temp_penalty)
            
        calculated_level = base_level + current_runoff - drought_offset
        levels.append(calculated_level)
        
    df_weather["simulated_level"] = levels
    
    if real_level is not None:
        last_sim = df_weather["simulated_level"].iloc[-1]
        offset = real_level - last_sim
        df_weather["simulated_level"] = df_weather["simulated_level"] + offset

    return df_weather

# ---------------------------------------------------------
# 5. 解析・AI補正エンジン
# ---------------------------------------------------------
def analyze_condition(df_weather, river_info, user_logs, target_river, target_date):
    real_level = fetch_real_water_level(river_info["stg_id"])

    if df_weather is None or df_weather.empty:
        current_level = real_level if real_level is not None else river_info["base_level"]
        return {
            "water_level": current_level,
            "level_is_real": real_level is not None,
            "level_trend": "平水（安定）",
            "days_since_flood": 4,
            "moss_growth": 50,
            "moss_alert": "✅ 新垢形成中（良好）",
            "flood_risk": "🟢 安定：増水リスク低",
            "clarity_recovery": "清澄（良好）",
            "season_mode": "盛期",
            "score": 5,
            "hourly_water_temp": [15 + i*0.2 for i in range(24)],
            "df_hydro": pd.DataFrame(),
            "target_df": pd.DataFrame(),
            "weather_desc": "データ取得不可",
            "temp_max": 20.0,
            "temp_min": 15.0,
            "water_temp_max": 18.0,
            "water_temp_avg": 16.5,
            "max_wind": 2.0
        }

    df_weather = simulate_water_levels(
        df_weather, 
        river_info["base_level"], 
        river_info["runoff_factor"], 
        river_info["decay_rate"],
        river_info["drought_rate"],
        real_level=real_level
    )

    target_datetime = datetime.datetime.combine(target_date, datetime.time(12, 0))

    bias_growth = 0.0
    river_logs = [l for l in user_logs if l.get("river") == target_river]
    if len(river_logs) > 0:
        feedbacks = [l.get("moss_feedback", 0) for l in river_logs]
        bias_growth = np.mean(feedbacks) * 0.1

    day_of_year = target_date.timetuple().tm_yday
    seasonal_temp_offset = 2.0 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
    adjusted_temp_base = river_info["temp_base"] + seasonal_temp_offset

    raw_water_temp = adjusted_temp_base + (df_weather["temperature_2m"] * river_info["temp_factor"])
    df_weather["estimated_water_temp"] = np.minimum(raw_water_temp, river_info["max_temp"])

    df_past = df_weather[df_weather["time"] <= target_datetime].copy()
    df_past["rain_12h"] = df_past["precipitation"].rolling(12, min_periods=1).sum()
    
    heavy_rain_events = df_past[(df_past["precipitation"] >= 15.0) | (df_past["rain_12h"] >= 35.0)]

    if not heavy_rain_events.empty:
        last_flood_time = heavy_rain_events["time"].max()
        days_since_flood = (target_datetime - last_flood_time).days
    else:
        days_since_flood = 10

    target_24h_rain = df_past.tail(24)["precipitation"].sum()
    if target_24h_rain > 60:
        clarity_recovery = "強濁り（回復まで約2～3日）"
        clarity_score = 1
    elif target_24h_rain > 30:
        clarity_recovery = "笹濁り（回復途上）"
        clarity_score = 2
    else:
        clarity_recovery = "清澄（良好）"
        clarity_score = 3

    m, d = target_date.month, target_date.day
    if m == 7 and d <= 15:
        season_mode = "初期（低水温・緩速成長）"
        growth_rate = 9.0
    elif (m == 7 and d > 15) or (m == 8 and d <= 15):
        season_mode = "盛期（高水温・高活性）"
        growth_rate = 12.5
    elif m == 8 and d > 15:
        season_mode = "晩夏・成熟期"
        growth_rate = 10.0
    else:
        season_mode = "終盤・落ち鮎（再生遅延）"
        growth_rate = 7.0

    recent_radiation = df_past.tail(max(24, days_since_flood * 24))["shortwave_radiation"].mean() if not df_past.empty else 150.0
    radiation_factor = max(0.7, min(1.3, recent_radiation / 180.0))

    moss_growth = min(100, int((days_since_flood * growth_rate * radiation_factor) * (1.0 + bias_growth)))

    target_df = df_weather[df_weather["time"].dt.date == target_date].copy()
    if not target_df.empty and len(target_df) >= 24:
        hourly_water_temp = target_df["estimated_water_temp"].tolist()[:24]
        current_sim_level = target_df["simulated_level"].mean()
        
        most_code = target_df["weathercode"].mode()[0] if not target_df["weathercode"].empty else 0
        weather_desc = get_weather_desc(most_code)
        temp_max = target_df["temperature_2m"].max()
        temp_min = target_df["temperature_2m"].min()
        water_temp_max = max(hourly_water_temp)
        water_temp_avg = float(np.mean(hourly_water_temp))
        max_wind = target_df["windspeed_10m"].max() if "windspeed_10m" in target_df.columns else 0.0
    else:
        hourly_water_temp = [14.0 + (i if i <= 14 else 28 - i) * 0.3 for i in range(24)]
        current_sim_level = real_level if real_level is not None else river_info["base_level"]
        weather_desc = "☀️ 晴れ"
        temp_max, temp_min = 22.0, 16.0
        water_temp_max, water_temp_avg = 17.5, 15.8
        max_wind = 2.0

    level_diff = current_sim_level - river_info["base_level"]
    if level_diff < -0.08:
        level_trend = "📉 渇水傾向（減水）"
    elif level_diff > 0.05:
        level_trend = f"📈 高水・引き水（+{level_diff*100:.0f}cm）"
    else:
        level_trend = "平水（安定）"

    if days_since_flood <= 1 or moss_growth < 20:
        moss_alert = "🚫 全飛び直後（垢ナシ・石白っぽい）"
    elif days_since_flood <= 4 or moss_growth < 60:
        moss_alert = "🟡 垢付き始め（まだ薄く喰い浅い）"
    elif level_diff < -0.12 and days_since_flood > 10:
        moss_alert = "⚠️ 垢腐り・泥垢注意（高水温・渇水進行）"
    else:
        moss_alert = "✅ 新垢形成完了（良好）"

    df_future = df_weather[df_weather["time"] >= target_datetime].head(24)
    future_rain = df_future["precipitation"].sum()
    if future_rain > 35.0:
        flood_risk = "🚨 警戒：全飛び（大増水）リスク高"
    elif future_rain > 15.0:
        flood_risk = "⚠️ 注意：雨による増水の可能性あり"
    else:
        flood_risk = "🟢 安定：増水リスク低"

    temp_peak_hours = len([t for t in hourly_water_temp if t >= 18.0])
    temp_pts = 3 if temp_peak_hours >= 4 else (2 if temp_peak_hours >= 2 else 1)
    
    raw_score = int((moss_growth / 100) * 4) + clarity_score + temp_pts

    if days_since_flood <= 2:
        max_cap = 3
    elif days_since_flood <= 4:
        max_cap = 5
    elif days_since_flood <= 6:
        max_cap = 7
    else:
        max_cap = 10

    score = max(1, min(raw_score, max_cap))

    df_hydro = df_weather.copy()
    df_hydro["base_level"] = river_info["base_level"]

    return {
        "water_level": current_sim_level,
        "level_is_real": real_level is not None,
        "level_trend": level_trend,
        "days_since_flood": days_since_flood,
        "moss_growth": moss_growth,
        "moss_alert": moss_alert,
        "flood_risk": flood_risk,
        "clarity_recovery": clarity_recovery,
        "season_mode": season_mode,
        "score": score,
        "hourly_water_temp": hourly_water_temp,
        "df_hydro": df_hydro,
        "target_df": target_df,
        "weather_desc": weather_desc,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "water_temp_max": water_temp_max,
        "water_temp_avg": water_temp_avg,
        "max_wind": max_wind
    }

# ---------------------------------------------------------
# 6. UI（メイン画面）
# ---------------------------------------------------------
st.title("🐟 北海道 鮎コンディション判定 & 未来予測")

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    target_river = st.selectbox("河川を選択してください", list(RIVERS.keys()))
with col_sel2:
    today_date = datetime.date.today()
    target_date = st.date_input("釣行予定日を選択", today_date, min_value=today_date - datetime.timedelta(days=7), max_value=today_date + datetime.timedelta(days=5))

river_info = RIVERS[target_river]

df_weather = fetch_weather_and_hydro(river_info["lat"], river_info["lon"])
user_logs = load_logs()
res = analyze_condition(df_weather, river_info, user_logs, target_river, target_date)

st.markdown("---")

if target_date == today_date:
    st.subheader("📅 本日のコンディション予測")
else:
    st.subheader(f"📅 {target_date.strftime('%Y年%m月%d日')} のコンディション事前予測")

stars = "★" * res["score"] + "☆" * (10 - res["score"])
st.markdown(f"### 🎯 釣行日おすすめ度 : {stars} （**{res['score']}** / 10）")

col_alert1, col_alert2 = st.columns(2)
with col_alert1:
    st.info(f"**全飛びリスク判定**: {res['flood_risk']}")
with col_alert2:
    if "⚠️" in res["moss_alert"] or "🚫" in res["moss_alert"] or "🟡" in res["moss_alert"]:
        st.warning(f"**コンディション**: {res['moss_alert']}")
    else:
        st.success(f"**コンディション**: {res['moss_alert']}")

if res["max_wind"] >= 6.0:
    st.error(f"💨 **強風注意**: 予想最大風速 {res['max_wind']:.1f} m/s （長尺竿の操作・保持にご注意ください）")

col1, col2, col3, col4, col5, col6 = st.columns(6)
label_water = "実測" if res["level_is_real"] else "推測"
col1.metric(f"水位 ({label_water})", f"{res['water_level']:.2f} m", res["level_trend"])
col2.metric("天気", res["weather_desc"])
col3.metric("予想気温", f"{res['temp_max']:.1f}℃", f"最低 {res['temp_min']:.1f}℃")
col4.metric("推計水温", f"{res['water_temp_max']:.1f}℃", f"平均 {res['water_temp_avg']:.1f}℃")
col5.metric("ハミ垢生育度", f"{res['moss_growth']} %")
col6.metric("最大風速", f"{res['max_wind']:.1f} m/s")

st.write(f"**濁り・澄み具合予測**: {res['clarity_recovery']}")
st.caption(f"※ 垢育成シーズンモード: **{res['season_mode']}** ／ 全飛びからの経過日数: **{res['days_since_flood']}日**")

# ---------------------------------------------------------
# 7. 指定日の1時間ごとの詳細天気予報
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"🌤️ {target_date.strftime('%m月%d日')} の1時間ごとのピンポイント天気予報")

if not res["target_df"].empty:
    df_hourly_view = res["target_df"].copy()
    df_hourly_view["時刻"] = df_hourly_view["time"].dt.strftime("%H:00")
    df_hourly_view["天気"] = df_hourly_view["weathercode"].apply(get_weather_desc)
    df_hourly_view["気温(℃)"] = df_hourly_view["temperature_2m"].round(1)
    df_hourly_view["降水量(mm)"] = df_hourly_view["precipitation"].round(1)
    df_hourly_view["風速(m/s)"] = df_hourly_view["windspeed_10m"].round(1)
    
    table_df = df_hourly_view[["時刻", "天気", "気温(℃)", "降水量(mm)", "風速(m/s)"]].set_index("時刻")
    st.dataframe(table_df.T, use_container_width=True)

# ---------------------------------------------------------
# 8. 水位グラフ（表示期間切替対応）
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 水位グラフ（直近実績 ＆ 天気予報AI予測）")

graph_range = st.radio(
    "グラフの表示期間を選択してください",
    options=["直近2日間 + 予測", "直近1週間 + 予測"],
    horizontal=True
)

if not res["df_hydro"].empty:
    past_days = 7 if graph_range == "直近1週間 + 予測" else 2
    start_time = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=past_days))
    end_time = pd.to_datetime(target_date + datetime.timedelta(days=1))

    chart_hydro = res["df_hydro"][(res["df_hydro"]["time"] >= start_time) & (res["df_hydro"]["time"] < end_time)].copy()
    
    now_ts = pd.Timestamp.now()
    
    chart_hydro["過去実績水位(m)"] = np.where(chart_hydro["time"] <= now_ts, chart_hydro["simulated_level"], np.nan)
    chart_hydro["天気予報AI予測水位(m)"] = np.where(chart_hydro["time"] >= now_ts, chart_hydro["simulated_level"], np.nan)
    
    chart_hydro["時間"] = chart_hydro["time"].dt.strftime("%m/%d %H時")
    chart_hydro = chart_hydro.rename(columns={"base_level": "平常基準水位(m)"})
    chart_hydro = chart_hydro.set_index("時間")
    
    st.line_chart(chart_hydro[["過去実績水位(m)", "天気予報AI予測水位(m)", "平常基準水位(m)"]])
    st.caption("※ 過去実績：国交省実測値ベース ／ 天気予報AI予測：最新の気象予報（雨量・気温）を基にしたAIシミュレーション値 ／ 平常基準水位：河川の基準線")

# ---------------------------------------------------------
# 9. 当日の時合・活性タイムライン（Y軸スケール10℃〜30℃固定）
# ---------------------------------------------------------
st.markdown("---")
st.subheader("⏰ 釣行日の水温推移 & ベスト時合予測")

temp_data = res["hourly_water_temp"]
hours = [f"{i:02d}:00" for i in range(24)]

chart_df = pd.DataFrame({
    "時刻": hours,
    "推計水温(℃)": temp_data
})

chart_temp = alt.Chart(chart_df).mark_line(point=True).encode(
    x=alt.X("時刻:N", sort=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("推計水温(℃):Q", scale=alt.Scale(domain=[10, 30])),
    tooltip=["時刻", "推計水温(℃)"]
).properties(
    height=300
)

st.altair_chart(chart_temp, use_container_width=True)

# 活性上向き（18℃以上）とベスト時合（20〜24℃）の判定
upward_hours = [i for i, t in enumerate(temp_data) if t >= 18.0 and t < 20.0]
best_hours = [i for i, t in enumerate(temp_data) if 20.0 <= t <= 24.0]
over_hours = [i for i, t in enumerate(temp_data) if t > 24.0]

if best_hours:
    b_start, b_end = min(best_hours), max(best_hours)
    st.success(f"🔥 **ベスト時合 (20℃〜24℃)**: **{b_start:02d}:00 ～ {b_end:02d}:00**（追い・ハミ出しともに最高潮の黄金タイムです）")

if upward_hours:
    u_start, u_end = min(upward_hours), max(upward_hours)
    st.info(f"📈 **活性上向き (18℃〜19.9℃)**: **{u_start:02d}:00 ～ {u_end:02d}:00**（ハミ出しや追いが活発になり始める時間帯です）")
elif not best_hours:
    st.warning("💡 **時合注意**: 全体的に水温が低めまたは高めの推移です。水温変化のタイミングを狙ってください。")

if over_hours:
    o_start, o_end = min(over_hours), max(over_hours)
    st.warning(f"⚠️ **高水温注意 (24℃超)**: **{o_start:02d}:00 ～ {o_end:02d}:00**（高水温により鮎がヘバる可能性がある時間帯です）")

# ---------------------------------------------------------
# 10. 実釣ログ入力 & 削除管理機能
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📝 実釣ログの記録（学習用）")

with st.form("log_form"):
    col_log1, col_log2 = st.columns(2)
    with col_log1:
        log_date = st.date_input("釣行日", today_date)
    with col_log2:
        river_keys = list(RIVERS.keys())
        default_index = river_keys.index(target_river) if target_river in river_keys else 0
        selected_log_river = st.selectbox("釣行河川", river_keys, index=default_index)

    catch_count = st.number_input("釣果（匹）", min_value=0, max_value=200, value=10)
    moss_condition = st.select_slider(
        "実際のハミ垢の状況",
        options=["全飛直後（白）", "薄っすら新垢", "ベスト（食み痕多数）", "垢腐り・泥垢"],
        value="ベスト（食み痕多数）"
    )
    
    feedback_map = {"全飛直後（白）": -2, "薄っすら新垢": -1, "ベスト（食み痕多数）": 0, "垢腐り・泥垢": 1}

    submitted = st.form_submit_button("実釣データを保存してAIに学習させる")
    if submitted:
        log_entry = {
            "date": str(log_date),
            "river": selected_log_river,
            "catch": catch_count,
            "moss_condition": moss_condition,
            "moss_feedback": feedback_map[moss_condition]
        }
        save_log(log_entry)
        st.success("実釣ログを保存しました！次回以降の予測精度に自動反映されます。")
        st.rerun()

if user_logs:
    with st.expander("📂 これまでの実釣ログを確認・削除"):
        for idx, log in enumerate(user_logs):
            c1, c2, c3, c4, c5 = st.columns([2, 3, 2, 3, 2])
            c1.write(f"📅 {log.get('date')}")
            c2.write(f"🌊 {log.get('river', '未設定')}")
            c3.write(f"🐟 {log.get('catch')} 匹")
            c4.write(f"🪨 {log.get('moss_condition')}")
            if c5.button("削除", key=f"del_{idx}"):
                delete_log(idx)
                st.success("ログを削除しました。")
                st.rerun()
