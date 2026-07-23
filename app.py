import streamlit as st

st.set_page_config(page_title="北海道 5河川 鮎コンディション判定", page_icon="🎣", layout="centered")

st.title("🎣 鮎コンディション判定")

# 河川データ設定
rivers = {
    "黒松内川": {"base_suigi": 1.40},
    "朱太川": {"base_suigi": 1.45},
    "昆布川": {"base_suigi": 43.50},
    "尻別川": {"base_suigi": 9.15},
    "天の川": {"base_suigi": 1.75}
}

selected_river = st.selectbox("河川を選択してください", list(rivers.keys()))

st.divider()

current_suigi = st.number_input(
    f"現在の水位 (m) [{selected_river}]", 
    min_value=0.0, max_value=50.0, 
    value=rivers[selected_river]["base_suigi"], 
    step=0.01, format="%.2f"
)

days_from_flood = st.slider("全飛び（大増水）からの経過日数", min_value=1, max_value=14, value=4)
weather_cond = st.selectbox("今日の天気・気温傾向", ["晴れ・高水温（新垢成長早）", "曇り・平年並み", "雨・増水あり"])

# スコア計算ロジック
# 1. 経過日数スコア (30%)
if days_from_flood <= 3:
    score_days = 2
elif days_from_flood == 4:
    score_days = 5
elif 5 <= days_from_flood <= 8:
    score_days = 9
else:
    score_days = 6

# 2. 縄張り・活性スコア (30%)
score_activity = 9 if days_from_flood >= 5 else 3

# 3. ポイント有効面積 (20%)
diff = abs(current_suigi - rivers[selected_river]["base_suigi"])
score_area = 9 if diff < 0.1 else (6 if diff < 0.25 else 3)

# 4. 水位安定度 (20%)
score_suigi = 8 if "晴れ" in weather_cond or "曇り" in weather_cond else 3

# 総合点算定 (10段階換算)
total_score = round((score_days * 0.3 + score_activity * 0.3 + score_area * 0.2 + score_suigi * 0.2), 1)

st.divider()

st.subheader(f"📌 {selected_river} のコンディション判定")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="評価スコア", value=f"{total_score} / 10")
with col2:
    st.metric(label="経過日数", value=f"{days_from_flood}日目")

if total_score >= 8.0:
    st.success("✨ **絶好調！** 新垢が完成し、縄張り意識が強い状態です。瀬のど真ん中からどこでも勝負になります。")
elif total_score >= 5.0:
    st.warning("⚠️ **追いが浅い・限定的。** チャラ瀬や浅場限定の釣りになります。丁寧な拾い釣りを推奨。")
else:
    st.error("❌ **条件不良。** 垢が未成熟、または増水・濁りの影響があります。")
