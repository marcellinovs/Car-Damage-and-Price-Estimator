import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from collections import Counter
from PIL import Image
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="Car Damage & Price ksgimis ",
    layout="centered"
)

CONF_THRESHOLD = 0.45
USD_TO_IDR = 16000
MAX_IMAGES = 5

# ==========================================
# PATH
# ==========================================
DAMAGE_MODEL_PATH = "runs/detect/damage_model_new5/weights/best.pt"

PRICE_MODEL_PATH = "price_model.pkl"

LE_MAKE_PATH = "le_make.pkl"
LE_MODEL_PATH = "le_model.pkl"

CAR_DATASET_PATH = "prices_dataset/car_prices.csv"

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():

    damage_model = YOLO(DAMAGE_MODEL_PATH)

    rf_model = joblib.load(PRICE_MODEL_PATH)

    le_make = joblib.load(LE_MAKE_PATH)
    le_model = joblib.load(LE_MODEL_PATH)

    return damage_model, rf_model, le_make, le_model


@st.cache_data
def load_dataset():

    df = pd.read_csv(
        CAR_DATASET_PATH,
        usecols=[
            "year",
            "make",
            "model",
            "mmr"
        ],
        low_memory=False
    )

    df.dropna(inplace=True)

    df["make"] = df["make"].astype(str)
    df["model"] = df["model"].astype(str)

    return df


# ==========================================
# INIT
# ==========================================
try:
    damage_model, rf_model, le_make, le_model = load_models()

    price_df = load_dataset()

except Exception as e:

    st.error(f"ERROR LOADING MODEL: {e}")

    st.stop()

# ==========================================
# FUNCTIONS
# ==========================================
def detect_damage(image):

    results = damage_model(
        image,
        conf=CONF_THRESHOLD,
        device="cpu"
    )

    damage_count = 0
    total_area = 0
    detected_classes = []

    img_np = np.array(image)

    h, w, _ = img_np.shape

    image_area = h * w

    class_names = damage_model.names

    for r in results:

        for box in r.boxes:

            damage_count += 1

            cls_id = int(box.cls[0])

            detected_classes.append(
                class_names.get(cls_id, str(cls_id))
            )

            x1, y1, x2, y2 = box.xyxy[0]

            area = float((x2 - x1) * (y2 - y1))

            total_area += area

    damage_ratio = 0

    if image_area > 0:
        damage_ratio = total_area / image_area

    return {
        "damage_count": damage_count,
        "damage_ratio": damage_ratio,
        "detected_classes": detected_classes,
        "results": results
    }


def estimate_condition(damage_count, damage_ratio):

    if damage_count == 0:
        return 5.0

    if damage_ratio < 0.03:
        return 4.0

    elif damage_ratio < 0.08:
        return 3.0

    elif damage_ratio < 0.15:
        return 2.0

    else:
        return 1.0


def estimate_repair_cost(damage_count, damage_ratio):

    base_cost = damage_count * 80

    if damage_ratio < 0.05:
        multiplier = 1

    elif damage_ratio < 0.15:
        multiplier = 1.5

    else:
        multiplier = 2

    return base_cost * multiplier


def safe_transform(encoder, value):

    value = str(value)

    if value in encoder.classes_:
        return encoder.transform([value])[0]

    return 0


def get_market_price(year, make):

    filtered = price_df[
        (price_df["year"] == year)
        &
        (price_df["make"].str.lower() == str(make).lower())
    ]

    if len(filtered) == 0:
        return float(price_df["mmr"].median())

    return float(filtered["mmr"].median())


def predict_price(
    year,
    make,
    model_name,
    condition,
    odometer,
    mmr
):

    make_encoded = safe_transform(le_make, make)

    model_encoded = safe_transform(le_model, model_name)

    features = [[
        year,
        make_encoded,
        model_encoded,
        condition,
        odometer,
        mmr
    ]]

    return rf_model.predict(features)[0]


def get_models(make):

    filtered = price_df[
        price_df["make"].str.lower() == str(make).lower()
    ]

    models = sorted(
        filtered["model"].dropna().unique()
    )

    if len(models) == 0:
        return sorted(le_model.classes_)

    return models


CONDITION_LABELS = {
    5.0: "Excellent",
    4.0: "Very Good",
    3.0: "Good",
    2.0: "Fair",
    1.0: "Poor",
}

# ==========================================
# UI
# ==========================================
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #111111;
}

/* Cards */
.card {
    background: #1C1C1C;
    border: 0.5px solid #2D2D2D;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Section labels */
.section-label {
    text-transform: uppercase;
    font-size: 11px;
    color: #5A5A58;
    letter-spacing: 0.07em;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Metric chips */
.metrics-row {
    display: flex;
    gap: 10px;
    margin-top: 4px;
}
.metric-chip {
    background: #252525;
    border-radius: 8px;
    padding: 12px 14px;
    flex: 1;
    text-align: center;
}
.metric-chip .chip-value {
    font-size: 22px;
    font-weight: 700;
    color: #EDEDEB;
    line-height: 1.2;
}
.metric-chip .chip-label {
    font-size: 11px;
    color: #5A5A58;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 3px;
}

/* Condition bar */
.bar-track {
    background: #252525;
    height: 8px;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 6px;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: #E24B4A;
}

/* Damage pills */
.damage-pill {
    display: inline-block;
    background: #2E1515;
    color: #F09595;
    border: 0.5px solid #5C2828;
    border-radius: 99px;
    padding: 4px 10px;
    font-size: 12px;
    margin: 3px 3px 3px 0;
}
.no-damage-pill {
    display: inline-block;
    background: #0D2820;
    color: #5DCAA5;
    border: 0.5px solid #1B4A3A;
    border-radius: 99px;
    padding: 4px 10px;
    font-size: 12px;
}

/* Thin divider */
.thin-divider {
    border: none;
    border-top: 0.5px solid #2D2D2D;
    margin: 16px 0;
}

/* Repair cost banner */
.repair-banner {
    background: #241A08;
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.repair-banner .banner-label {
    font-size: 13px;
    color: #C4986A;
}
.repair-banner .banner-value {
    font-size: 18px;
    font-weight: 700;
    color: #F0C070;
}

/* Price cards */
.price-cards {
    display: flex;
    gap: 12px;
}
.price-card {
    flex: 1;
    background: #1C1C1C;
    border: 0.5px solid #2D2D2D;
    border-radius: 12px;
    padding: 16px;
}
.price-card.highlight {
    background: #0D2820;
    border-color: #1B4A3A;
}
.price-card .card-label {
    font-size: 11px;
    color: #5A5A58;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.price-card .card-usd {
    font-size: 22px;
    font-weight: 700;
    color: #EDEDEB;
}
.price-card .card-idr {
    font-size: 12px;
    color: #5A5A58;
    margin-top: 3px;
}

/* Image caption label */
.img-label {
    font-size: 11px;
    color: #5A5A58;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
    text-align: center;
}

/* Analyze button */
div.stButton > button {
    background-color: #EDEDEB !important;
    color: #111111 !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #FFFFFF !important;
    color: #111111 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Car Damage & Price Estimator")

# ==========================================
# VEHICLE PHOTOS CARD
# ==========================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Vehicle photos</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload vehicle images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

images = []

if uploaded_files:

    if len(uploaded_files) > MAX_IMAGES:

        st.warning(f"Maximum {MAX_IMAGES} images allowed.")

        uploaded_files = uploaded_files[:MAX_IMAGES]

    cols = st.columns(min(len(uploaded_files), 5))

    for idx, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")

        image = image.resize((640, 640))

        images.append(image)

        with cols[idx % 5]:

            st.image(image, use_container_width=True)

            st.markdown(
                f'<div class="img-label">Image {idx + 1}</div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# VEHICLE DETAILS CARD
# ==========================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Vehicle details</div>', unsafe_allow_html=True)

year = st.number_input(
    "Year",
    min_value=1980,
    max_value=2015,
    value=2012
)

make = st.selectbox(
    "Brand",
    sorted(le_make.classes_)
)

available_models = get_models(make)

model_name = st.selectbox(
    "Model",
    available_models
)

odometer = st.number_input(
    "Odometer",
    min_value=0,
    max_value=500000,
    value=80000
)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PROCESS
# ==========================================
if uploaded_files:

    if st.button("Analyze Vehicle"):

        total_damage_count = 0

        damage_ratios = []

        all_classes = []

        all_results = []

        with st.spinner("Analyzing vehicle..."):

            for image in images:

                result = detect_damage(image)

                total_damage_count += result["damage_count"]

                damage_ratios.append(
                    result["damage_ratio"]
                )

                all_classes.extend(
                    result["detected_classes"]
                )

                all_results.append(
                    result["results"]
                )

        avg_damage_ratio = 0

        if len(damage_ratios) > 0:

            avg_damage_ratio = float(
                np.mean(damage_ratios)
            )

        condition = estimate_condition(
            total_damage_count,
            avg_damage_ratio
        )

        repair_cost = estimate_repair_cost(
            total_damage_count,
            avg_damage_ratio
        )

        mmr = get_market_price(
            year,
            make
        )

        base_price = predict_price(
            year,
            make,
            model_name,
            condition,
            odometer,
            mmr
        )

        final_price = max(
            base_price - repair_cost,
            0
        )

        condition_label = CONDITION_LABELS.get(condition, "Unknown")
        bar_width = int(condition / 5 * 100)

        # ==========================================
        # RESULT CARD
        # ==========================================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Detection result</div>', unsafe_allow_html=True)

        # Block A — Metrics row
        damage_pct = f"{avg_damage_ratio * 100:.1f}%"
        condition_str = f"{int(condition)} / 5"

        st.markdown(f"""
<div class="metrics-row">
  <div class="metric-chip">
    <div class="chip-value">{total_damage_count}</div>
    <div class="chip-label">Damage count</div>
  </div>
  <div class="metric-chip">
    <div class="chip-value">{damage_pct}</div>
    <div class="chip-label">Damage ratio</div>
  </div>
  <div class="metric-chip">
    <div class="chip-value">{condition_str}</div>
    <div class="chip-label">Condition</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Block B — Condition bar
        st.markdown(f"""
<div style="margin-top:16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:12px;color:#5A5A58;text-transform:uppercase;letter-spacing:0.06em;">Condition score</span>
    <span style="font-size:13px;font-weight:600;color:#F08080;">{condition_label}</span>
  </div>
  <div class="bar-track">
    <div class="bar-fill" style="width:{bar_width}%;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Block C — Damage tags
        class_counts = Counter(all_classes)

        if class_counts:
            pills_html = "".join(
                f'<span class="damage-pill">{cls} &times; {cnt}</span>'
                if cnt > 1
                else f'<span class="damage-pill">{cls}</span>'
                for cls, cnt in class_counts.items()
            )
        else:
            pills_html = '<span class="no-damage-pill">No damage detected</span>'

        st.markdown(f"""
<div style="margin-top:16px;">
  <div class="section-label">Detected damage types</div>
  <div>{pills_html}</div>
</div>
""", unsafe_allow_html=True)

        # Divider
        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

        # Block D — Repair cost banner
        st.markdown(f"""
<div class="repair-banner">
  <span class="banner-label">Estimated repair cost</span>
  <span class="banner-value">${int(repair_cost):,}</span>
</div>
""", unsafe_allow_html=True)

        # Block E — Price cards
        base_idr = f"Rp {int(base_price * USD_TO_IDR):,}"
        final_idr = f"Rp {int(final_price * USD_TO_IDR):,}"

        st.markdown(f"""
<div class="price-cards">
  <div class="price-card">
    <div class="card-label">Base selling price</div>
    <div class="card-usd">${int(base_price):,}</div>
    <div class="card-idr">{base_idr}</div>
  </div>
  <div class="price-card highlight">
    <div class="card-label">Final selling price</div>
    <div class="card-usd">${int(final_price):,}</div>
    <div class="card-idr">{final_idr}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ==========================================
        # DETECTION VISUALIZATION
        # ==========================================
        st.markdown(
            '<div class="section-label" style="margin-top:24px;">Detection visualization</div>',
            unsafe_allow_html=True
        )

        for idx, results in enumerate(all_results):

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-label">Image {idx + 1}</div>',
                unsafe_allow_html=True
            )

            annotated = results[0].plot()

            st.image(
                annotated,
                use_container_width=True
            )

            st.markdown('</div>', unsafe_allow_html=True)
