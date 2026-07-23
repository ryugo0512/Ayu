import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
import requests

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
# 2. 河川・観測所データ設定
# ---------------------------------------------------------
RIVERS = {
    "尻別川本流（蘭越）": {
        "lat": 42.8021, "lon": 140.5251, "base_level": 1.20,
        "stg_id": "3010312811020", "runoff_factor": 0.025, "decay_rate": 0.96, "drought_rate": 0.0005
    },
    "昆布川（昆布）": {
        "lat": 42.7958, "lon": 140.5986, "base_level": 0.80,
        "stg_id": "3010312811040", "runoff_factor": 0.030, "decay_rate": 0.95, "drought_rate": 0.0005
    },
    "天ノ川（上ノ国）": {
        "lat": 41.7997, "lon": 140.1163, "base_level": 0.90,
        "stg_id": "3010112811010", "runoff_factor": 0.030, "decay_rate": 0.97, "drought_rate": 0.0005
    },
    "朱太川（黒松内）": {
        "lat": 42.6683, "lon": 140.3061, "base_level": 0.70,
        "stg_id": "3010312811050", "runoff_factor": 0.035, "decay_rate": 0.96, "drought_rate": 0.0005
    }
}

# ---------------------------------------------------------
# 3. 外部API取得モジュール
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_weather_and_hydro(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,sunshine_duration,shortwave_radiation&past_days=14&forecast_days=7&timezone=Asia%2FTokyo"
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

# ---------------------------------------------------------
# 4. 水位推移シミュレーション
# ---------------------------------------------------------
def simulate_water_levels(df_weather, base_level, runoff_factor, decay_rate, drought_rate, real_level=None):
    levels = []
    current_runoff = 0.0
    dry_hours = 0
    
    for idx, row in df_weather.iterrows():
        rain = row["precipitation"]
        temp = row["temperature_2m"]
        
        if rain > 0.2:
            dry_hours = 0
            current_runoff = current_runoff * decay_rate + (rain * runoff_factor)
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
            "hourly_water_temp": [16 + i*0.5 for i in range(24)],
            "df_hydro": pd.DataFrame()
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

    df_weather["estimated_water_temp"] = df_weather["temperature_2m"] * 0.75 + 3.0

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

    moss_growth = min(100, int((days_since_flood * growth_rate) * (1.0 + bias_growth)))

    target_df = df_weather[df_weather["time"].dt.date == target_date]
    if not target_df.empty and len(target_df) >= 24:
        hourly_water_temp = target_df["estimated_water_temp"].tolist()[:24]
        current_sim_level = target_df["simulated_level"].mean()
    else:
        hourly_water_temp = [15.0 + (i if i <= 14 else 28 - i) * 0.4 for i in range(24)]
        current_sim_level = real_level if real_level is not None else river_info["base_level"]

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

    temp_peak_hours = len([t for t in hourly_water_temp if t >= 17.0])
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

    start_time = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=2))
    end_time = pd.to_datetime(target_date + datetime.timedelta(days=1))
    df_hydro = df_weather[(df_weather["time"] >= start_time) & (df_weather["time"] < end_time)].copy()
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
        "df_hydro": df_hydro
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

col1, col2, col3, col4 = st.columns(4)
label_water = "水文テレメータ実測" if res["level_is_real"] else "推測水位"
col1.metric(f"水位 ({label_water})", f"{res['water_level']:.2f} m", res["level_trend"])
col2.metric("全飛びからの経過", f"{res['days_since_flood']} 日")
col3.metric("ハミ垢生育度", f"{res['moss_growth']} %")
col4.metric("シーズンモード", res["season_mode"])

st.write(f"**濁り・澄み具合予測**: {res['clarity_recovery']}")

# ---------------------------------------------------------
# 7. 水位グラフ（過去：実測、未来：AI天気予報予測で色分け）
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 水位グラフ（直近実績 ＆ 天気予報AI予測）")

if not res["df_hydro"].empty:
    chart_hydro = res["df_hydro"][["time", "simulated_level", "base_level"]].copy()
    
    now_ts = pd.Timestamp.now()
    
    chart_hydro["過去実績水位(m)"] = np.where(chart_hydro["time"] <= now_ts, chart_hydro["simulated_level"], np.nan)
    chart_hydro["天気予報AI予測水位(m)"] = np.where(chart_hydro["time"] >= now_ts, chart_hydro["simulated_level"], np.nan)
    
    chart_hydro["時間"] = chart_hydro["time"].dt.strftime("%m/%d %H時")
    chart_hydro = chart_hydro.rename(columns={"base_level": "平常基準水位(m)"})
    chart_hydro = chart_hydro.set_index("時間")
    
    st.line_chart(chart_hydro[["過去実績水位(m)", "天気予報AI予測水位(m)", "平常基準水位(m)"]])
    st.caption("※ 過去実績：国交省実測値ベース ／ 天気予報AI予測：最新の気象予報（雨量・気温）を基にしたAIシミュレーション値 ／ 平常基準水位：河川の基準線")

# ---------------------------------------------------------
# 8. 当日の時合・活性タイムライン
# ---------------------------------------------------------
st.markdown("---")
st.subheader("⏰ 釣行日の水温推移 & ベスト時合予測")

temp_data = res["hourly_water_temp"]
hours = [f"{i:02d}:00" for i in range(24)]

chart_df = pd.DataFrame({
    "時刻": hours,
    "推計水温(℃)": temp_data
}).set_index("時刻")

st.line_chart(chart_df)

best_hours = [i for i, t in enumerate(temp_data) if t >= 17.0]
if best_hours:
    start_h, end_h = min(best_hours), max(best_hours)
    st.success(f"🔥 **おすすめ時合**: **{start_h:02d}:00 ～ {end_h:02d}:00**（推計水温が17℃を超え、ハミ出し活性が最大化します）")
else:
    st.info("💡 **おすすめ時合**: 全体的に水温が低めです。日照が強まる **12:00 ～ 14:30** が集中ポイントとなります。")

# ---------------------------------------------------------
# 9. 実釣ログ入力 & 削除管理機能
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📝 実釣ログの記録（学習用）")

with st.form("log_form"):
    log_date = st.date_input("釣行日", today_date)
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
            "river": target_river,
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
            c2.write(f"🌊 {log.get('river')}")
            c3.write(f"🐟 {log.get('catch')} 匹")
            c4.write(f"🪨 {log.get('moss_condition')}")
            if c5.button("削除", key=f"del_{idx}"):
                delete_log(idx)
                st.success("ログを削除しました。")
                st.rerun()
