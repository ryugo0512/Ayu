import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="水位予測システム", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/ryugo0512/ayu_prediction_system/main/data/water_levels.json"

LOCATIONS = {
    "rankoshi": {"lat": 42.79, "lon": 140.47},
    "niseko": {"lat": 42.80, "lon": 140.68},
    "kutchan": {"lat": 42.90, "lon": 140.76},
    "kimobetsu": {"lat": 42.79, "lon": 140.92}
}

RIVERS = {
    "尻別川本流": {"base_level": 9.08},
    "昆布川": {"base_level": 43.58},
    "天ノ川": {"base_level": 1.60},
    "朱太川": {"base_level": 1.44}
}

def get_jst_now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))

@st.cache_data(ttl=3600)
def load_water_data():
    try:
        res = requests.get(DATA_URL, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception:
        return {}

@st.cache_data(ttl=3600)
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

def train_and_predict(df_past, df_future, base_level):
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

    now = get_jst_now()
    future_times = [now + datetime.timedelta(hours=i) for i in range(1, 25)]
    pred_levels = []
    
    last_level = df_past["water_level"].iloc[-1] if not df_past.empty else base_level
    
    for f_time in future_times:
        f_time_str = f_time.strftime("%Y-%m-%d %H:00:00")
        
        r_ran = r_nis = r_kut = r_kim = 0.0
        if not df_future.empty:
            rain_row = df_future[df_future["time"] == f_time_str]
            if not rain_row.empty:
                r_ran = rain_row["rain_rankoshi"].values[0]
                r_nis = rain_row["rain_niseko"].values[0]
                r_kut = rain_row["rain_kutchan"].values[0]
                r_kim = rain_row["rain_kimobetsu"].values[0]
        
        X_pred = pd.DataFrame([[last_level, r_ran, r_nis, r_kut, r_kim]], columns=features)
        
        if len(train_df) > 10:
            next_level = model.predict(X_pred)[0]
        else:
            next_level = last_level * 0.99 + (r_ran+r_nis+r_kut+r_kim)*0.01 + model.intercept_
        
        pred_levels.append(next_level)
        last_level = next_level

    df_pred = pd.DataFrame({
        "time": future_times,
        "predicted_level": pred_levels
    })
    return df_pred

def main():
    st.title("水位予測システム")
    
    river_name = st.selectbox("対象河川を選択", list(RIVERS.keys()))
    base_level = RIVERS[river_name]["base_level"]
    
    data_json = load_water_data()
    df_future_rain = fetch_future_rain()
    
    df_past = pd.DataFrame()
    if river_name in data_json and data_json[river_name]:
        df_past = pd.DataFrame(data_json[river_name])
        df_past["time"] = pd.to_datetime(df_past["timestamp"])
        df_past = df_past.sort_values("time").reset_index(drop=True)
    
    if df_past.empty:
        st.warning("過去データがありません。")
        return

    df_pred = train_and_predict(df_past, df_future_rain, base_level)
    
    st.markdown("---")
    st.subheader("水位グラフ")
    graph_range = st.radio("グラフ表示期間", ["直近2日間", "直近1週間", "直近2週間"], horizontal=True)
    
    days_map = {"直近2日間": 2, "直近1週間": 7, "直近2週間": 14}
    past_days = days_map.get(graph_range, 2)
    start_time = get_jst_now() - datetime.timedelta(days=past_days)
    
    df_past_disp = df_past[df_past["time"] >= start_time.replace(tzinfo=None)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_past_disp["time"], 
        y=[base_level]*len(df_past_disp), 
        mode="lines", 
        name="基準水位線(m)", 
        line=dict(color="navy")
    ))
    
    fig.add_trace(go.Scatter(
        x=df_past_disp["time"], 
        y=df_past_disp["water_level"], 
        mode="lines", 
        name="過去水位(m)", 
        line=dict(color="dodgerblue")
    ))
    
    fig.add_trace(go.Scatter(
        x=df_pred["time"], 
        y=df_pred["predicted_level"], 
        mode="lines", 
        name="予測水位(m)", 
        line=dict(color="red")
    ))
    
    fig.update_layout(xaxis_title="時間", yaxis_title="水位", height=500)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
