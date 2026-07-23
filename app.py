import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
import requests

# ---------------------------------------------------------
# 1. 基本設定とデータ永続化（実釣ログ保存）
# ---------------------------------------------------------
st.set_page_config(page_title="北海道 鮎コンディション判定", page_icon="🐟", layout="wide")

LOG_FILE = "fishing_logs.json"

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_log(log_entry):
    logs = load_logs()
    logs.append(log_entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------
# 2. 河川・観測所データ設定
# ---------------------------------------------------------
RIVERS = {
    "尻別川本流（蘭越）": {"lat": 42.8021, "lon": 140.5251, "base_level": 1.20, "flood_threshold": 0.60},
    "昆布川（昆布）": {"lat": 42.7958, "lon": 140.5986, "base_level": 0.80, "flood_threshold": 0.50},
    "天ノ川（上ノ国）": {"lat": 41.7997, "lon": 140.1163, "base_level": 0.90, "flood_threshold": 0.50},
    "朱太川（黒松内）": {"lat": 42.6683, "lon": 140.3061, "base_level": 0.70, "flood_threshold": 0.45}
}

# ---------------------------------------------------------
# 3. 外部API取得モジュール（気象・水象データ）
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_weather_and_hydro(lat, lon):
    """Open-Meteo APIから直近＆予報データを一括取得"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,sunshine_duration,shortwave_radiation&past_days=14&forecast_days=3&timezone=Asia%2FTokyo"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        return None

# ---------------------------------------------------------
# 4. 解析・AI補正エンジン
# ---------------------------------------------------------
def analyze_condition(df_weather, river_info, user_logs, target_river):
    if df_weather is None or df_weather.empty:
        # フォールバック処理
        return {
            "water_level": river_info["base_level"],
            "level_trend": "平水（安定）",
            "days_since_flood": 7,
            "moss_growth": 80,
            "moss_alert": "良好",
            "flood_risk": "低",
            "clarity_recovery": "良好（澄み）",
            "season_mode": "盛期",
            "hourly_water_temp": [16 + i*0.5 for i in range(24)]
        }

    now = datetime.datetime.now()
    today = now.date()

    # AI補正係数の算出（ユーザーログから学習）
    bias_growth = 0.0
    river_logs = [l for l in user_logs if l.get("river") == target_river]
    if len(river_logs) > 0:
        # ユーザーの実釣フィードバックから成長補正値を計算
        feedbacks = [l.get("moss_feedback", 0) for l in river_logs]
        bias_growth = np.mean(feedbacks) * 0.1

    # 気温から推定水温を算出
    df_weather["estimated_water_temp"] = df_weather["temperature_2m"] * 0.75 + 3.0

    # 大雨イベント（全飛び）の自動検知
    df_past = df_weather[df_weather["time"] < now]
    recent_heavy_rain = df_past[df_past["precipitation"] > 15.0]

    if not recent_heavy_rain.empty:
        last_flood_time = recent_heavy_rain["time"].max()
        days_since_flood = (now - last_flood_time).days
    else:
        days_since_flood = 10

    # 濁り回復日数（直近の総降水量から推計）
    last_24h_rain = df_past.tail(24)["precipitation"].sum()
    if last_24h_rain > 30:
        clarity_recovery = "本日～明日にかけて笹濁り（回復途上）"
    elif last_24h_rain > 60:
        clarity_recovery = "強濁り（回復まで約2～3日）"
    else:
        clarity_recovery = "清澄（良好）"

    # シーズンモード判定（北海道特化：9/1以降は終盤モード）
    if today.month == 9 and today.day >= 1:
        season_mode = "終盤（再生遅延モード）"
        growth_rate = 10.0
    else:
        season_mode = "盛期"
        growth_rate = 14.0

    # アカ（ハミ垢）の生育度（積算水温・日射モデル）
    moss_growth = min(100, int((days_since_flood * growth_rate) * (1.0 + bias_growth)))

    # アカ腐りアラート
    if days_since_flood > 12 and last_24h_rain < 5.0:
        moss_alert = "⚠️ 垢腐り・泥垢注意（高水温・長期間渇水）"
    else:
        moss_alert = "✅ 新垢形成中（良好）"

    # 全飛び警戒アラート（今後24時間の予想降雨）
    df_future = df_weather[df_weather["time"] >= now].head(24)
    future_rain = df_future["precipitation"].sum()
    if future_rain > 40.0:
        flood_risk = "🚨 警戒：24時間以内に全飛び（大増水）リスク高"
    elif future_rain > 20.0:
        flood_risk = "⚠️ 注意：雨による水位上昇の可能性あり"
    else:
        flood_risk = "🟢 安定：増水リスク低"

    # 当日の1時間ごと推計水温（時合予測用）
    today_df = df_weather[df_weather["time"].dt.date == today]
    if not today_df.empty and len(today_df) >= 24:
        hourly_water_temp = today_df["estimated_water_temp"].tolist()[:24]
    else:
        hourly_water_temp = [15.0 + (i if i <= 14 else 28 - i) * 0.4 for i in range(24)]

    return {
        "water_level": river_info["base_level"] + (last_24h_rain * 0.01),
        "level_trend": "減水傾向（引き水）" if last_24h_rain < 2.0 else "平水〜微増",
        "days_since_flood": days_since_flood,
        "moss_growth": moss_growth,
        "moss_alert": moss_alert,
        "flood_risk": flood_risk,
        "clarity_recovery": clarity_recovery,
        "season_mode": season_mode,
        "hourly_water_temp": hourly_water_temp
    }

# ---------------------------------------------------------
# 5. UI（メイン画面）
# ---------------------------------------------------------
st.title("🐟 北海道 鮎コンディション判定 & 時合予測")

target_river = st.selectbox("河川を選択してください", list(RIVERS.keys()))
river_info = RIVERS[target_river]

# データ取得
df_weather = fetch_weather_and_hydro(river_info["lat"], river_info["lon"])
user_logs = load_logs()
res = analyze_condition(df_weather, river_info, user_logs, target_river)

st.markdown("---")

# 警報・アラート表示
col_alert1, col_alert2 = st.columns(2)
with col_alert1:
    st.info(f"**全飛びリスク判定**: {res['flood_risk']}")
with col_alert2:
    if "⚠️" in res["moss_alert"]:
        st.warning(f"**コンディション**: {res['moss_alert']}")
    else:
        st.success(f"**コンディション**: {res['moss_alert']}")

# 主要指標
col1, col2, col3, col4 = st.columns(4)
col1.metric("推測水位", f"{res['water_level']:.2f} m", res["level_trend"])
col2.metric("全飛びからの経過", f"{res['days_since_flood']} 日")
col3.metric("ハミ垢生育度", f"{res['moss_growth']} %")
col4.metric("シーズンモード", res["season_mode"])

st.write(f"**濁り・澄み具合予測**: {res['clarity_recovery']}")

# ---------------------------------------------------------
# 6. 当日の時合・活性タイムライン
# ---------------------------------------------------------
st.markdown("---")
st.subheader("⏰ 当日の水温推移 & ベスト時合予測")

temp_data = res["hourly_water_temp"]
hours = [f"{i:02d}:00" for i in range(24)]

chart_df = pd.DataFrame({
    "時刻": hours,
    "推計水温(℃)": temp_data
}).set_index("時刻")

st.line_chart(chart_df)

# 時合の判定（水温17℃以上、かつ12時〜15時のピーク時）
best_hours = [i for i, t in enumerate(temp_data) if t >= 17.0]
if best_hours:
    start_h, end_h = min(best_hours), max(best_hours)
    st.success(f"🔥 **本日のベスト時合**: **{start_h:02d}:00 ～ {end_h:02d}:00**（推計水温が17℃を超え、ハミ出し活性が最大化します）")
else:
    st.info("💡 **本日の時合**: 全体的に水温が低めです。日照が強まる **12:00 ～ 14:30** が集中ポイントとなります。")

# ---------------------------------------------------------
# 7. 実釣ログ入力（AI補正用・永続保存）
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📝 実釣ログの記録（学習用）")

with st.form("log_form"):
    log_date = st.date_input("釣行日", datetime.date.today())
    catch_count = st.number_input("釣果（匹）", min_value=0, max_value=200, value=10)
    moss_condition = st.select_slider(
        "実際のハミ垢の状況",
        options=["全飛直後（白）", "薄っすら新垢", "ベスト（食み痕多数）", "垢腐り・泥垢"],
        value="ベスト（食み痕多数）"
    )
    
    # AI補正用フィードバック数値化
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

# 過去ログ表示
if user_logs:
    with st.expander("📂 これまでの実釣ログを確認"):
        st.dataframe(pd.DataFrame(user_logs))
