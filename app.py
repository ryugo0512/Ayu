import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import os
import json

st.set_page_config(page_title="鮎釣り総合予測システム", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/ryugo0512/ayu_prediction_system/main/data/water_levels.json"
LOG_FILE = "fishing_logs.json"
WATER_TEMP_LOG_FILE = "water_temp_logs.json"

LOCATIONS = {
    "rankoshi": {"lat": 42.79, "lon": 140.47},
    "niseko": {"lat": 42.80, "lon": 140.68},
    "kutchan": {"lat": 42.90, "lon": 140.76},
    "kimobetsu": {"lat": 42.79, "lon": 140.92}
}

RIVERS = {
    "尻別川本流": {"lat": 42.79, "lon": 140.47, "base_level": 9.08},
    "昆布川": {"lat": 42.79, "lon": 140.53, "base_level": 43.58},
    "天ノ川": {"lat": 41.88, "lon": 140.13, "base_level": 1.60},
    "朱太川": {"lat": 42.64, "lon": 140.32, "base_level": 1.44}
}

def get_jst_now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))

@st.cache_data(ttl=1800)
def load_water_data():
    try:
        res = requests.get(DATA_URL, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception:
        return {}

@st.cache_data(ttl=1800)
def fetch_weather_and_temp(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,shortwave_radiation&timezone=Asia%2FTokyo&forecast_days=2"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "temp": data["hourly"]["temperature_2m"],
            "rad": data["hourly"]["shortwave_radiation"]
        })
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_future_rain():
    future_rain = {}
    for loc, coords in LOCATIONS.items():
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&hourly=precipitation&timezone=Asia%2FTokyo&forecast_days=2"
        try:
            res = requests.get(url, timeout=10)
            data = res.json()
            df = pd.DataFrame({
                "time": pd.to_datetime(data["hourly"]["time"]),
                f"rain_{loc}": data["hourly"]["precipitation"]
            })
            future_rain[loc] = df
        except Exception:
            pass
    
    if not future_rain:
        return pd.DataFrame()
    
    merged_df = future_rain["rankoshi"]
    for loc in ["niseko", "kutchan", "kimobetsu"]:
        if loc in future_rain:
            merged_df = pd.merge(merged_df, future_rain[loc], on="time", how="outer")
    return merged_df

def load_logs(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_log(file_path, data):
    logs = load_logs(file_path)
    logs.append(data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def predict_water_temp(df_weather):
    if df_weather.empty:
        return pd.DataFrame()
    df = df_weather.copy()
    df["predicted_water_temp"] = df["temp"] * 0.7 + (df["rad"] / 100) * 0.5 + 4.0
    return df

def calculate_recommendation(current_level, base_level, pred_level, current_temp):
    diff = current_level - base_level
    pred_diff = pred_level - current_level

    if -0.2 <= diff <= 0.15:
        water_score = 5
        water_msg = "平水〜好水位"
    elif 0.15 < diff <= 0.4:
        water_score = 4
        water_msg = "やや増水（笹濁り注意）"
    elif -0.4 <= diff < -0.2:
        water_score = 3
        water_msg = "渇水ぎみ"
    elif diff > 0.4:
        water_score = 1
        water_msg = "増水（高水位・危険）"
    else:
        water_score = 1
        water_msg = "大渇水"

    if pred_diff > 0.2:
        water_score = max(1, water_score - 2)
        water_msg += " ※今後増水予測あり"

    if 18.0 <= current_temp <= 24.0:
        temp_score = 5
        temp_msg = "適性水温（高活性期待）"
    elif 16.0 <= current_temp < 18.0 or 24.0 < current_temp <= 26.0:
        temp_score = 3
        temp_msg = "やや低温/高温"
    else:
        temp_score = 1
        temp_msg = "活性低い可能性"

    total_score = round(water_score * 0.6 + temp_score * 0.4)
    stars = "★" * total_score + "☆" * (5 - total_score)
    
    return stars, f"【水位】{water_msg} / 【水温】{temp_msg}"

def train_and_predict_water_level(df_past, df_future, base_level):
    for col in ["rain_rankoshi", "rain_niseko", "rain_kutchan", "rain_kimobetsu"]:
        if col not in df_past.columns:
            df_past[col] = 0.0
    df_past = df_past.fillna(0.0)

    df_past["prev_level"] = df_past["water_level"].shift(1)
    train_df = df_past.dropna().copy()

    features = ["prev_level", "rain_rankoshi", "rain_niseko", "rain_kutchan", "rain_kimobetsu"]
    
    model = LinearRegression()
    if len(train_df) > 10:
        X_train = train_df[features]
        y_train = train_df["water_level"]
        model.fit(X_train, y_train)
    else:
        model.coef_ = np.array([0.99, 0.01, 0.01, 0.01, 0.01])
        model.intercept_ = base_level * (1 - 0.99)

    if not df_past.empty:
        last_time = df_past["time"].iloc[-1]
        last_level = df_past["water_level"].iloc[-1]
    else:
        last_time = get_jst_now().replace(tzinfo=None)
        last_level = base_level

    future_times = [last_time + datetime.timedelta(hours=i) for i in range(1, 25)]
    pred_times = [last_time] + future_times
    pred_levels = [last_level]
    
    current_val = last_level
    for f_time in future_times:
        f_time_str = f_time.strftime("%Y-%m-%d %H:00:00")
        
        r_ran = r_nis = r_kut = r_kim = 0.0
        if not df_future.empty:
            rain_row = df_future[df_future["time"] == f_time_str]
            if not rain_row.empty:
                r_ran = rain_row["rain_rankoshi"].values[0] if "rain_rankoshi" in rain_row.columns else 0.0
                r_nis = rain_row["rain_niseko"].values[0] if "rain_niseko" in rain_row.columns else 0.0
                r_kut = rain_row["rain_kutchan"].values[0] if "rain_kutchan" in rain_row.columns else 0.0
                r_kim = rain_row["rain_kimobetsu"].values[0] if "rain_kimobetsu" in rain_row.columns else 0.0
        
        X_pred = pd.DataFrame([[current_val, r_ran, r_nis, r_kut, r_kim]], columns=features)
        
        if len(train_df) > 10:
            next_level = model.predict(X_pred)[0]
        else:
            next_level = current_val * 0.99 + (r_ran+r_nis+r_kut+r_kim)*0.01 + model.intercept_
        
        pred_levels.append(next_level)
        current_val = next_level

    df_pred = pd.DataFrame({
        "time": pred_times,
        "predicted_level": pred_levels
    })
    return df_pred

def main():
    st.title("鮎釣り総合予測システム")
    
    river_name = st.selectbox("対象河川を選択", list(RIVERS.keys()))
    river_info = RIVERS[river_name]
    
    data_json = load_water_data()
    df_weather = fetch_weather_and_temp(river_info["lat"], river_info["lon"])
    df_future_rain = fetch_future_rain()
    df_temp_pred = predict_water_temp(df_weather)
    
    df_past = pd.DataFrame()
    if river_name in data_json and data_json[river_name]:
        df_past = pd.DataFrame(data_json[river_name])
        df_past["time"] = pd.to_datetime(df_past["timestamp"], errors="coerce")
        df_past = df_past.dropna(subset=["time"])
        df_past["time"] = df_past["time"].dt.tz_localize(None)
        df_past = df_past.sort_values("time").reset_index(drop=True)

    current_level = df_past["water_level"].iloc[-1] if not df_past.empty else river_info["base_level"]
    current_temp = df_temp_pred["predicted_water_temp"].iloc[0] if not df_temp_pred.empty else 18.0
    
    df_pred = pd.DataFrame()
    if not df_past.empty:
        df_pred = train_and_predict_water_level(df_past, df_future_rain, river_info["base_level"])
        pred_24h_level = df_pred["predicted_level"].iloc[-1]
    else:
        pred_24h_level = current_level

    stars, rec_msg = calculate_recommendation(current_level, river_info["base_level"], pred_24h_level, current_temp)

    # 1. サマリーダッシュボード
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("現在水位", f"{current_level:.2f} m", f"基準比: {current_level - river_info['base_level']:+.2f} m")
    m_col2.metric("24時間後予測水位", f"{pred_24h_level:.2f} m", f"現在比: {pred_24h_level - current_level:+.2f} m")
    m_col3.metric("推定水温", f"{current_temp:.1f} ℃")
    m_col4.metric("釣行オススメ度", stars)
    
    st.info(f"**コンディション診断:** {rec_msg}")

    # 2. 水位グラフ（連続線描画）
    st.markdown("---")
    st.subheader("水位グラフ（重回帰分析モデル）")
    
    if not df_past.empty:
        graph_range = st.radio("グラフ表示期間", ["直近2日間", "直近1週間", "直近2週間"], horizontal=True, index=1)
        days_map = {"直近2日間": 2, "直近1週間": 7, "直近2週間": 14}
        past_days = days_map.get(graph_range, 7)
        
        last_time = df_past["time"].iloc[-1]
        start_time = last_time - datetime.timedelta(days=past_days)
        df_past_disp = df_past[df_past["time"] >= start_time]
        
        fig_water = go.Figure()
        
        fig_water.add_trace(go.Scatter(
            x=[df_past_disp["time"].iloc[0], df_pred["time"].iloc[-1]], 
            y=[river_info["base_level"], river_info["base_level"]], 
            mode="lines", 
            name="基準水位線(m)", 
            line=dict(color="navy", dash="dash")
        ))
        
        fig_water.add_trace(go.Scatter(
            x=df_past_disp["time"], 
            y=df_past_disp["water_level"], 
            mode="lines+markers", 
            name="過去水位(m)", 
            line=dict(color="dodgerblue", width=2),
            marker=dict(size=4)
        ))
        
        fig_water.add_trace(go.Scatter(
            x=df_pred["time"], 
            y=df_pred["predicted_level"], 
            mode="lines", 
            name="予測水位(m)", 
            line=dict(color="red", width=2)
        ))
        
        fig_water.update_layout(
            xaxis_title="時間", 
            yaxis_title="水位(m)", 
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_water, use_container_width=True)
    else:
        st.warning("水位蓄積データがありません。")

    # 3. 水温予測グラフ
    st.markdown("---")
    st.subheader("水温予測グラフ")
    if not df_temp_pred.empty:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=df_temp_pred["time"], 
            y=df_temp_pred["predicted_water_temp"], 
            mode="lines+markers", 
            name="予測水温(℃)",
            line=dict(color="orange", width=2)
        ))
        fig_temp.update_layout(xaxis_title="時間", yaxis_title="水温 (℃)", height=350, hovermode="x unified")
        st.plotly_chart(fig_temp, use_container_width=True)

    # 4. ログ入力フォーム
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("水温ログ入力")
        with st.form("water_temp_form"):
            temp_date = st.date_input("日付", datetime.date.today())
            temp_val = st.number_input("実測水温 (℃)", min_value=0.0, max_value=35.0, value=18.0, step=0.1)
            submit_temp = st.form_submit_button("水温ログを保存")
            if submit_temp:
                log_data = {"date": str(temp_date), "river": river_name, "water_temp": temp_val}
                save_log(WATER_TEMP_LOG_FILE, log_data)
                st.success("水温ログを保存した。")

    with col2:
        st.subheader("釣果ログ入力")
        with st.form("fishing_log_form"):
            fish_date = st.date_input("日付 ", datetime.date.today())
            count = st.number_input("尾数", min_value=0, max_value=200, value=10)
            memo = st.text_input("メモ (ポイント、ハリス等)")
            submit_fish = st.form_submit_button("釣果ログを保存")
            if submit_fish:
                log_data = {"date": str(fish_date), "river": river_name, "count": count, "memo": memo}
                save_log(LOG_FILE, log_data)
                st.success("釣果ログを保存した。")

    # 5. ログ表示タブ
    st.markdown("---")
    st.subheader("保存ログ確認")
    tab1, tab2 = st.tabs(["水温履歴", "釣果履歴"])
    
    with tab1:
        temp_logs = load_logs(WATER_TEMP_LOG_FILE)
        if temp_logs:
            st.dataframe(pd.DataFrame(temp_logs), use_container_width=True)
        else:
            st.info("水温ログはまだありません。")

    with tab2:
        fish_logs = load_logs(LOG_FILE)
        if fish_logs:
            st.dataframe(pd.DataFrame(fish_logs), use_container_width=True)
        else:
            st.info("釣果ログはまだありません。")

if __name__ == "__main__":
    main()
