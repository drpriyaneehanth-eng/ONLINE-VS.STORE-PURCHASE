"""
train_model.py
--------------
Inspects the dataset, trains multiple classifiers, selects the best one,
and saves the full sklearn pipeline to models/model.pkl.
"""

import os
import pathlib
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape : {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ── 2. Auto-identify target ───────────────────────────────────────────────────
TARGET = "shopping_preference"
assert TARGET in df.columns, f"Target column '{TARGET}' not found."
print(f"\nTarget column : {TARGET}")
print(f"  Value counts:\n{df[TARGET].value_counts()}")

# ── 3. Separate features and target ──────────────────────────────────────────
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ── 4. Identify column types ──────────────────────────────────────────────────
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\nNumeric features  ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ── 5. Build preprocessing pipeline ──────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

# ── 6. Encode target ──────────────────────────────────────────────────────────
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nTarget classes: {list(label_encoder.classes_)}")

# ── 7. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 8. Model comparison ───────────────────────────────────────────────────────
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
}

print("\n── Model Comparison (5-fold CV on training set) ──")
results = {}
for name, clf in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    results[name] = cv_scores.mean()
    print(f"  {name:30s}  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}  (CV accuracy = {results[best_name]:.4f})")

# ── 9. Train best model on full training set ──────────────────────────────────
best_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("classifier", candidates[best_name])]
)
best_pipeline.fit(X_train, y_train)

# ── 10. Evaluate on test set ──────────────────────────────────────────────────
y_pred = best_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {test_acc:.4f}")
print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    )
)

# ── 11. Save pipeline + metadata ─────────────────────────────────────────────
artifact = {
    "pipeline": best_pipeline,
    "label_encoder": label_encoder,
    "feature_names": X.columns.tolist(),
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "model_name": best_name,
    "test_accuracy": test_acc,
    "target_column": TARGET,
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(artifact, f)

print(f"\nModel saved → {MODEL_PATH}")
print("Done ✓")
