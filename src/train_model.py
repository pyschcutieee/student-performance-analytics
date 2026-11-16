import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "student_performance_cleaned.csv"
MODEL_PATH = BASE_DIR / "dashboard" / "student_risk_model_v2.pkl"

df = pd.read_csv(DATA_PATH)

features = [
    "gender",
    "age",
    "address",
    "G1",
    "G2",
    "health",
    "studytime",
    "failures",
    "absentees"
]

X = df[features]
y = df["risk_level"]

categorical_features = ["gender", "address"]
numeric_features = [
    "age", "G1", "G2", "health",
    "studytime", "failures", "absentees"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numeric", "passthrough", numeric_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, MODEL_PATH)

print("\nModel trained successfully!")
print("Saved model:", MODEL_PATH)
