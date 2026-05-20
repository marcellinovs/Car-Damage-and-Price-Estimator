import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="Car Damage & Price Estimator",
    layout="centered"
)

CONF_THRESHOLD = 0.45
USD_TO_IDR = 16000
MAX_IMAGES = 5

# ==========================================
# PATH
# ==========================================
DAMAGE_MODEL_PATH = "runs/detect/damage_model_new2/weights/best.pt"

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

# ==========================================
# UI
# ==========================================
st.title("Car Damage & Price Estimator")

uploaded_files = st.file_uploader(
    "Upload gambar mobil",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.subheader("Vehicle Information")

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

# ==========================================
# PROCESS
# ==========================================
if uploaded_files:

    if len(uploaded_files) > MAX_IMAGES:

        st.warning(
            f"Maksimal {MAX_IMAGES} gambar"
        )

        uploaded_files = uploaded_files[:MAX_IMAGES]

    images = []

    st.subheader("Uploaded Images")

    cols = st.columns(min(len(uploaded_files), 5))

    for idx, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")

        image = image.resize((640, 640))

        images.append(image)

        with cols[idx % 5]:

            st.image(
                image,
                caption=f"Image {idx+1}",
                use_container_width=True
            )

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

        # ==========================================
        # RESULT
        # ==========================================
        st.subheader("Detection Result")

        st.write(
            f"Detected Damage Count: {total_damage_count}"
        )

        st.write(
            f"Vehicle Condition: {condition}"
        )

        st.write(
            f"Damage Types: {all_classes if all_classes else 'No Damage'}"
        )

        st.write(
            f"Estimated Repair Cost: ${int(repair_cost):,}"
        )

        st.write(
            f"Estimated Selling Price: ${int(final_price):,}"
        )

        st.write(
            f"Estimated Selling Price (IDR): Rp {int(final_price * USD_TO_IDR):,}"
        )

        st.subheader("Detection Visualization")

        for idx, results in enumerate(all_results):

            st.write(f"Image {idx+1}")

            annotated = results[0].plot()

            st.image(
                annotated,
                use_container_width=True
            )