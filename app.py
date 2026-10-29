"""Streamlit dashboard for three-phase voltage unbalance severity."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
DATA_PATH = ROOT / "data" / "unseen_test_data.csv"
FEATURES = ["Vr_V", "Vy_V", "Vb_V", "Average_V", "Unbalance_Percent"]

st.set_page_config(page_title="Three-Phase Voltage Unbalance AI", page_icon="⚡", layout="wide")
st.title("⚡ AI-Based Three-Phase Voltage Unbalance Detection")
st.caption("Educational prototype • Synthetic data • Not a protection relay")

def engineer_features(vr, vy, vb):
    avg = (vr + vy + vb) / 3.0
    if avg <= 0:
        return avg, float("nan")
    unb = max(abs(vr-avg), abs(vy-avg), abs(vb-avg)) / abs(avg) * 100.0
    return avg, unb

def classify_by_threshold(unb):
    if unb <= 1.0:
        return "Balanced"
    if unb <= 2.0:
        return "Mildly Unbalanced"
    return "Severely Unbalanced"

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python train_model.py` in the project folder first.")
    st.stop()

model = joblib.load(MODEL_PATH)
tab1, tab2, tab3 = st.tabs(["Live input prediction", "Unseen test data", "Project notes"])

with tab1:
    st.subheader("Enter three phase-to-neutral RMS voltages")
    c1, c2, c3 = st.columns(3)
    with c1: vr = st.number_input("Phase R voltage (V)", min_value=1.0, max_value=500.0, value=230.0, step=0.5)
    with c2: vy = st.number_input("Phase Y voltage (V)", min_value=1.0, max_value=500.0, value=230.0, step=0.5)
    with c3: vb = st.number_input("Phase B voltage (V)", min_value=1.0, max_value=500.0, value=230.0, step=0.5)
    avg, unb = engineer_features(vr, vy, vb)
    row = pd.DataFrame([[vr, vy, vb, avg, unb]], columns=FEATURES)
    prediction = model.predict(row)[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average voltage", f"{avg:.2f} V")
    m2.metric("Voltage unbalance", f"{unb:.3f}%")
    m3.metric("AI severity", prediction)
    m4.metric("Threshold reference", classify_by_threshold(unb))
    st.info("The threshold reference is shown separately from the AI prediction so you can compare them.")
    chart = pd.DataFrame({"Phase": ["R", "Y", "B"], "Voltage (V)": [vr, vy, vb]})
    st.bar_chart(chart.set_index("Phase"))
    if prediction != classify_by_threshold(unb):
        st.warning("AI prediction differs from the configured threshold label. Review model/data and engineering limits.")

with tab2:
    st.subheader("Run the model on separate unseen readings")
    unseen = pd.read_csv(DATA_PATH)
    if st.button("Predict unseen dataset"):
        unseen["AI_Prediction"] = model.predict(unseen[FEATURES])
        st.dataframe(unseen.head(50), use_container_width=True)
        st.caption(f"Showing first 50 of {len(unseen)} synthetic unseen samples.")
        st.bar_chart(unseen["AI_Prediction"].value_counts())
    else:
        st.write(f"Dataset ready: {len(unseen)} synthetic unseen samples.")
        st.dataframe(unseen.head(10), use_container_width=True)

with tab3:
    st.markdown("""
**Configured illustrative thresholds**
- Balanced: unbalance <= 1.0%
- Mildly unbalanced: 1.0% < unbalance <= 2.0%
- Severely unbalanced: unbalance > 2.0%

**Metric used:** maximum absolute phase deviation from average phase voltage divided by average phase voltage, multiplied by 100.

**Limitations:** The dataset is synthetic and the threshold scheme is for a classroom demonstration. Real-world monitoring should use calibrated, isolated measurement hardware and the engineering definition/limits specified by the applicable standard or utility. Do not connect a microcontroller directly to mains voltage.
""")
