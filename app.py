"""
app.py
------
Streamlit web application for the Online vs Store Shopping Preference predictor.
Run with:  streamlit run app.py
"""

import pathlib
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopping Preference Predictor",
    page_icon="🛒",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

artifact = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 Shopping Preference Predictor")
st.markdown(
    """
Predict whether a customer prefers **Online**, **In-Store**, or a **Hybrid**
shopping experience based on their demographic and behavioural profile.
"""
)

if artifact is None:
    st.error(
        "Model file not found at `models/model.pkl`.  "
        "Please run **`python train_model.py`** first."
    )
    st.stop()

pipeline = artifact["pipeline"]
label_encoder = artifact["label_encoder"]
model_name = artifact["model_name"]
test_accuracy = artifact["test_accuracy"]
numeric_cols = artifact["numeric_cols"]
categorical_cols = artifact["categorical_cols"]

st.success(
    f"✅ Model loaded: **{model_name}** | Test accuracy: **{test_accuracy:.2%}**"
)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Enter Customer Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=90, value=35, step=1)
    monthly_income = st.number_input(
        "Monthly Income (₹)", min_value=5000, max_value=500000, value=50000, step=1000
    )
    daily_internet_hours = st.slider(
        "Daily Internet Hours", min_value=0.0, max_value=16.0, value=5.0, step=0.1
    )
    smartphone_usage_years = st.slider(
        "Smartphone Usage (years)", min_value=0, max_value=20, value=6, step=1
    )
    social_media_hours = st.slider(
        "Social Media Hours / day", min_value=0.0, max_value=12.0, value=2.0, step=0.1
    )
    online_payment_trust_score = st.slider(
        "Online Payment Trust Score (1–10)", min_value=1, max_value=10, value=5
    )
    tech_savvy_score = st.slider(
        "Tech-Savvy Score (1–10)", min_value=1, max_value=10, value=5
    )
    monthly_online_orders = st.slider(
        "Monthly Online Orders", min_value=0, max_value=60, value=10, step=1
    )
    monthly_store_visits = st.slider(
        "Monthly Store Visits", min_value=0, max_value=30, value=5, step=1
    )
    avg_online_spend = st.number_input(
        "Avg Monthly Online Spend (₹)", min_value=0, max_value=300000, value=20000, step=500
    )
    avg_store_spend = st.number_input(
        "Avg Monthly Store Spend (₹)", min_value=0, max_value=300000, value=20000, step=500
    )

with col2:
    discount_sensitivity = st.slider(
        "Discount Sensitivity (1–10)", min_value=1, max_value=10, value=5
    )
    return_frequency = st.slider(
        "Return Frequency (times/month)", min_value=0, max_value=15, value=2, step=1
    )
    avg_delivery_days = st.slider(
        "Avg Delivery Days", min_value=1, max_value=15, value=4, step=1
    )
    delivery_fee_sensitivity = st.slider(
        "Delivery Fee Sensitivity (1–10)", min_value=1, max_value=10, value=5
    )
    free_return_importance = st.slider(
        "Free Return Importance (1–10)", min_value=1, max_value=10, value=5
    )
    product_availability_online = st.slider(
        "Product Availability Online (1–10)", min_value=1, max_value=10, value=5
    )
    impulse_buying_score = st.slider(
        "Impulse Buying Score (1–10)", min_value=1, max_value=10, value=5
    )
    need_touch_feel_score = st.slider(
        "Need to Touch/Feel Score (1–10)", min_value=1, max_value=10, value=5
    )
    brand_loyalty_score = st.slider(
        "Brand Loyalty Score (1–10)", min_value=1, max_value=10, value=5
    )
    environmental_awareness = st.slider(
        "Environmental Awareness (1–10)", min_value=1, max_value=10, value=5
    )
    time_pressure_level = st.slider(
        "Time Pressure Level (1–10)", min_value=1, max_value=10, value=5
    )
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    city_tier = st.selectbox("City Tier", options=["Tier 1", "Tier 2", "Tier 3"])

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Shopping Preference", use_container_width=True, type="primary"):
    input_dict = {
        "age": age,
        "monthly_income": monthly_income,
        "daily_internet_hours": daily_internet_hours,
        "smartphone_usage_years": smartphone_usage_years,
        "social_media_hours": social_media_hours,
        "online_payment_trust_score": online_payment_trust_score,
        "tech_savvy_score": tech_savvy_score,
        "monthly_online_orders": monthly_online_orders,
        "monthly_store_visits": monthly_store_visits,
        "avg_online_spend": avg_online_spend,
        "avg_store_spend": avg_store_spend,
        "discount_sensitivity": discount_sensitivity,
        "return_frequency": return_frequency,
        "avg_delivery_days": avg_delivery_days,
        "delivery_fee_sensitivity": delivery_fee_sensitivity,
        "free_return_importance": free_return_importance,
        "product_availability_online": product_availability_online,
        "impulse_buying_score": impulse_buying_score,
        "need_touch_feel_score": need_touch_feel_score,
        "brand_loyalty_score": brand_loyalty_score,
        "environmental_awareness": environmental_awareness,
        "time_pressure_level": time_pressure_level,
        "gender": gender,
        "city_tier": city_tier,
    }

    input_df = pd.DataFrame([input_dict])

    pred_encoded = pipeline.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # Probability scores
    proba = pipeline.predict_proba(input_df)[0]
    classes = label_encoder.classes_

    st.divider()
    st.subheader("Prediction Result")

    emoji_map = {"Online": "💻", "Store": "🏪", "Hybrid": "🔄"}
    color_map = {"Online": "#1f77b4", "Store": "#2ca02c", "Hybrid": "#ff7f0e"}

    st.markdown(
        f"""
        <div style="
            background-color:{color_map.get(pred_label, '#444')};
            color:white;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:1.8em;
            font-weight:bold;
        ">
        {emoji_map.get(pred_label, '🛍️')} Predicted Preference: {pred_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Probability Breakdown")
    proba_df = pd.DataFrame(
        {"Shopping Preference": classes, "Probability": proba}
    ).sort_values("Probability", ascending=False)

    for _, row in proba_df.iterrows():
        st.progress(
            float(row["Probability"]),
            text=f"{row['Shopping Preference']} — {row['Probability']:.1%}",
        )

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        """
This app uses a machine-learning model trained on the
**Online vs Store Shopping Dataset** (≈ 11,789 records).

**Problem type:** Multiclass Classification

**Target classes:**
- 🏪 Store
- 💻 Online
- 🔄 Hybrid

**Training pipeline:**
1. Numeric features → StandardScaler
2. Categorical features → OneHotEncoder
3. Best classifier selected via 5-fold CV

---
Run `python train_model.py` to retrain the model.
        """
    )
    if artifact:
        st.metric("Best Model", model_name)
        st.metric("Test Accuracy", f"{test_accuracy:.2%}")
