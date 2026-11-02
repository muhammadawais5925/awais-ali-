# Product Category Prediction (Random Forest)

This project trains a Random Forest classifier to predict a product's category from its price, quantity, rating, and calculated revenue.

## Files
- `app.py` — Streamlit app
- `model_pipeline.py` — trains and saves the model
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
python model_pipeline.py
streamlit run app.py
```

The training data in this project is a small illustrative sample (10 rows), so predictions are for demonstration and are not suitable for real-world decisions without a larger, representative dataset.
