"""
train_model.py
==============
Trains both models from the notebook, saves to models/ folder.
Run this if model pkl files are missing:
    python train_model.py
"""
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transaction_generator import generate_transactions

os.makedirs("models", exist_ok=True)

print("=" * 55)
print(" CreditIQ — Model Training")
print("=" * 55)

# ── Step 1: Generate training data ────────────────────────────────
print("\n[1/4] Generating training transactions (200 accounts)...")
records = []
for i in range(200):
    acc  = f"TRAIN{str(i).zfill(6)}"
    data = generate_transactions(acc, months=18)
    for txn in data["transactions"]:
        records.append({
            "account_number":  acc,
            "amount":          float(txn["amount"]),
            "type":            1 if txn["type"].upper() == "CREDIT" else 0,
            "is_late_payment": 1 if txn.get("is_late_payment") else 0,
            "balance_after":   float(txn.get("balance_after") or 0),
            "description":     str(txn.get("merchant","")).upper(),
            "category":        txn.get("category","OTHER"),
        })

df = pd.DataFrame(records)
print(f"      {len(df):,} transactions | {df['account_number'].nunique()} accounts")
print(f"      Categories: {sorted(df['category'].unique())}")

# ── Step 2: Category classifier (mirrors notebook cells 5-9) ─────
print("\n[2/4] Training category classifier (TF-IDF + LogisticRegression)...")
X_text = df["description"]
y_cat  = df["category"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vec      = vectorizer.fit_transform(X_text)
le         = LabelEncoder()
y_enc      = le.fit_transform(y_cat)

X_tr, X_te, y_tr, y_te = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)
cat_model = LogisticRegression(max_iter=1000)
cat_model.fit(X_tr, y_tr)
acc_cat = accuracy_score(y_te, cat_model.predict(X_te))
print(f"      Accuracy: {acc_cat*100:.2f}%")

# ── Step 3: Risk model (mirrors notebook cells 11-17) ────────────
print("\n[3/4] Training risk model (RandomForest)...")
features = df.groupby("account_number").agg({
    "amount":          ["sum", "mean", "std"],
    "type":            "sum",
    "is_late_payment": "sum",
    "balance_after":   ["mean", "min"],
})
features.columns = ["_".join(col) for col in features.columns]
features.reset_index(inplace=True)
features["amount_std"] = features["amount_std"].fillna(0)

# Risk label — exactly as in the notebook
features["risk_label"] = (
    (features["is_late_payment_sum"] > 3) |
    (features["amount_sum"]          < 5000) |
    (features["balance_after_min"]   < 1000)
).astype(int)

FEATURE_COLS = [
    "amount_sum", "amount_mean", "amount_std",
    "type_sum", "is_late_payment_sum",
    "balance_after_mean", "balance_after_min",
]

X_f = features[FEATURE_COLS]
y_r = features["risk_label"]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_f, y_r, test_size=0.25, random_state=42, stratify=y_r)

risk_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
risk_model.fit(X_tr2, y_tr2)
acc_risk = accuracy_score(y_te2, risk_model.predict(X_te2))
print(f"      Accuracy: {acc_risk*100:.2f}%")

# Feature importances
imp = pd.Series(risk_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"      Top features: {list(imp.index[:3])}")

# ── Step 4: Save all models ───────────────────────────────────────
print("\n[4/4] Saving models...")
joblib.dump(cat_model,  "models/category_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(le,         "models/label_encoder.pkl")
joblib.dump(risk_model, "models/risk_model.pkl")

# Verify
for f in ["category_model.pkl","tfidf_vectorizer.pkl","label_encoder.pkl","risk_model.pkl"]:
    size = os.path.getsize(f"models/{f}") / 1024
    print(f"      models/{f}  ({size:.0f} KB)")

print("\n" + "="*55)
print(f"  Category Classifier : {acc_cat*100:.2f}% accuracy")
print(f"  Risk Model          : {acc_risk*100:.2f}% accuracy")
print("  All models saved to models/")
print("="*55)
