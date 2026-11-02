import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA = {
    "Category": ["Electronics", "Clothing", "Electronics", "Home & Kitchen",
                 "Clothing", "Electronics", "Books", "Home & Kitchen",
                 "Books", "Electronics"],
    "Price": [1200, 50, 800, 300, 80, 1500, 20, 150, 15, 600],
    "Quantity": [1, 2, 1, 1, 3, 1, 5, 2, 10, 2],
    "Rating": [4.5, 4.0, 4.8, 3.9, 4.2, 4.7, 4.1, 4.0, 4.5, 4.3],
}

FEATURES = ["Price", "Quantity", "Rating", "Revenue"]
MODEL_PATH = Path(__file__).with_name("random_forest_model.pkl")


def main():
    df = pd.DataFrame(DATA)
    df["Revenue"] = df["Price"] * df["Quantity"]

    X, y = df[FEATURES], df["Category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(classification_report(y_test, predictions, zero_division=0))

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
