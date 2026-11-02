from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("random_forest_model.pkl")
FEATURES = ["Price", "Quantity", "Rating", "Revenue"]

st.set_page_config(page_title="Product Category Predictor", page_icon="🛍️")
st.title("Product Category Predictor")
st.write("Enter product details to predict its category.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python model_pipeline.py` first.")
    st.stop()

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

price = st.number_input("Price", min_value=0.0, value=100.0, step=1.0)
quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
rating = st.slider("Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
revenue = price * quantity

if st.button("Predict category"):
    sample = pd.DataFrame(
        [[price, quantity, rating, revenue]],
        columns=FEATURES,
    )
    st.success(f"Predicted category: {model.predict(sample)[0]}")
    st.caption(f"Calculated revenue: {revenue:.2f}")
