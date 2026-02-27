"""
predict.py  —  Exact pipeline from model1.ipynb, extended with rich features.

Two models (both trained from notebook logic):
  1. category_model  (TF-IDF + LogisticRegression)   — 100% accuracy
     description text -> category label

  2. risk_model  (GradientBoostingRegressor)
     18 per-account features -> CIBIL score 300-900
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

BASE_FEATURES = [
    "amount_sum","amount_mean","amount_std",
    "type_sum","is_late_payment_sum",
    "balance_after_mean","balance_after_min",
]
RICH_FEATURES = [
    "income_monthly","savings_rate","sip_monthly","emi_monthly","dti_ratio",
    "late_rate","bal_volatility","digital_rate","has_sip","has_emi","total_txns",
]
FEATURE_COLS = BASE_FEATURES + RICH_FEATURES

_models = {}

def _load():
    if not _models:
        _models["cat"]  = joblib.load(MODEL_DIR / "category_model.pkl")
        _models["vec"]  = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
        _models["le"]   = joblib.load(MODEL_DIR / "label_encoder.pkl")
        _models["risk"] = joblib.load(MODEL_DIR / "risk_model.pkl")
        fc_path = MODEL_DIR / "feature_cols.pkl"
        _models["fc"]   = joblib.load(fc_path) if fc_path.exists() else FEATURE_COLS


def predict_category(description: str) -> str:
    _load()
    vec  = _models["vec"].transform([description.upper()])
    pred = _models["cat"].predict(vec)
    return _models["le"].inverse_transform(pred)[0]


def predict_cibil(transactions: list) -> dict:
    _load()
    fc = _models["fc"]

    if not transactions:
        return {"cibil_score": 300, "risk_prob": 1.0, **_score_meta(300)}

    rows = []
    for t in transactions:
        rows.append({
            "account_number":  t.get("account_number", "ACC"),
            "amount":          float(t.get("amount", 0) or 0),
            "type":            1 if str(t.get("type","")).upper() == "CREDIT" else 0,
            "is_late_payment": 1 if str(t.get("is_late_payment","NO")).upper() in ("YES","1","TRUE") else 0,
            "balance_after":   float(t.get("balance_after", 0) or 0),
            "description":     str(t.get("merchant","")).upper(),
            "category":        t.get("category", "OTHER"),
        })

    df     = pd.DataFrame(rows)
    months = 18

    agg = df.groupby("account_number").agg({
        "amount":          ["sum","mean","std"],
        "type":            "sum",
        "is_late_payment": "sum",
        "balance_after":   ["mean","min"],
    })
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.reset_index(inplace=True)
    agg["amount_std"] = agg["amount_std"].fillna(0)

    cr = df[df["type"] == 1]
    db = df[df["type"] == 0]

    inc_mo = cr["amount"].sum() / months
    exp_mo = db["amount"].sum() / months
    sav_r  = max(0.0, (inc_mo - exp_mo) / inc_mo) if inc_mo > 0 else 0.0
    sip_mo = db[db["category"] == "SIP"]["amount"].sum() / months
    emi_mo = db[db["category"] == "EMI_LOAN"]["amount"].sum() / months
    dti    = emi_mo / inc_mo if inc_mo > 0 else 0.0

    bal_avg = float(df["balance_after"].mean())
    bal_cv  = float(df["balance_after"].std() or 0) / (bal_avg + 1)

    bills   = df[df["category"].isin(["BILL_PAYMENT","EMI_LOAN","INSURANCE","RENT"])]
    late_r  = float(df["is_late_payment"].sum()) / max(len(bills), 1)
    dig_r   = float(df["description"].str.startswith(("UPI","IMPS")).sum()) / max(len(df), 1)

    agg["income_monthly"] = inc_mo
    agg["savings_rate"]   = sav_r
    agg["sip_monthly"]    = sip_mo
    agg["emi_monthly"]    = emi_mo
    agg["dti_ratio"]      = dti
    agg["late_rate"]      = late_r
    agg["bal_volatility"] = bal_cv
    agg["digital_rate"]   = dig_r
    agg["has_sip"]        = 1 if sip_mo > 0 else 0
    agg["has_emi"]        = 1 if emi_mo > 0 else 0
    agg["total_txns"]     = len(df)

    for col in fc:
        if col not in agg.columns:
            agg[col] = 0.0

    X     = agg[fc].astype(float).fillna(0)
    raw   = float(_models["risk"].predict(X)[0])
    score = max(300, min(900, int(round(raw))))
    risk_p = round(1.0 - (score - 300) / 600, 4)

    return {"cibil_score": score, "risk_prob": risk_p, **_score_meta(score)}


def _score_meta(score: int) -> dict:
    if score >= 750:
        return dict(grade="Excellent", risk_level="Very Low Risk",  risk_color="#27AE60", percentile=88)
    if score >= 650:
        return dict(grade="Good",      risk_level="Low Risk",       risk_color="#82CA9D", percentile=70)
    if score >= 550:
        return dict(grade="Fair",      risk_level="Medium Risk",    risk_color="#F5A623", percentile=48)
    if score >= 450:
        return dict(grade="Poor",      risk_level="High Risk",      risk_color="#E67E22", percentile=28)
    return dict(grade="Very Poor",     risk_level="Very High Risk", risk_color="#E74C3C", percentile=10)


def score_components(transactions: list) -> list:
    if not transactions:
        return []

    months  = 18
    debits  = [t for t in transactions if str(t.get("type","")).upper() == "DEBIT"]
    credits = [t for t in transactions if str(t.get("type","")).upper() == "CREDIT"]
    late    = [t for t in transactions if str(t.get("is_late_payment","NO")).upper() in ("YES","1","TRUE")]
    bills   = [t for t in transactions if t.get("category","") in
               ("BILL_PAYMENT","EMI_LOAN","RENT","INSURANCE")]

    avg_income  = sum(t["amount"] for t in credits) / months or 1
    avg_expense = sum(t["amount"] for t in debits)  / months
    savings_r   = max(0.0, (avg_income - avg_expense) / avg_income)

    late_rate  = len(late) / max(len(bills), 1)
    on_time_r  = 1.0 - late_rate

    sip_amt    = sum(t["amount"] for t in debits if t.get("category") == "SIP")
    emi_amt    = sum(t["amount"] for t in debits if t.get("category") == "EMI_LOAN")
    has_ins    = any(t.get("category") == "INSURANCE" for t in transactions)
    has_sip    = sip_amt > 0
    has_stocks = any(t.get("category") == "STOCKS" for t in transactions)
    dti        = (emi_amt / months) / avg_income if avg_income else 0

    def grade(r):
        if r >= .85: return "A"
        if r >= .70: return "B"
        if r >= .55: return "C"
        if r >= .40: return "D"
        return "F"

    pb  = on_time_r
    is_ = min(1.0, avg_income / 80000) * 0.7 + (0.3 if avg_income > 20000 else 0.0)
    sv  = min(1.0, savings_r / 0.40)
    inv = min(1.0, (0.4 if has_sip else 0) + (0.3 if has_stocks else 0) + (0.3 if has_ins else 0))
    dm  = max(0.0, 1.0 - dti / 0.6) if dti > 0 else 0.85

    return [
        dict(name="Payment Behavior",    score=round(pb  * 100, 1), weight=35, grade=grade(pb),
             insight=(f"{len(late)} late payment(s) out of {len(bills)} bills."
                      if late else "All bills paid on time."),
             improvement="Set up auto-debit for all recurring bills and EMIs."),
        dict(name="Income Stability",    score=round(is_ * 100, 1), weight=25, grade=grade(is_),
             insight=f"Avg monthly income ₹{avg_income:,.0f}.",
             improvement="Add a second income source — freelance or investment returns."),
        dict(name="Savings & Cushion",   score=round(sv  * 100, 1), weight=20, grade=grade(sv),
             insight=f"Saving {savings_r*100:.0f}% of income each month.",
             improvement="Target 20%+ savings rate. Open a recurring deposit."),
        dict(name="Investment Behavior", score=round(inv * 100, 1), weight=10, grade=grade(inv),
             insight=("Active SIP + insurance detected." if has_sip and has_ins
                      else "No SIP or insurance detected."),
             improvement="Start a ₹500/month SIP to signal financial maturity."),
        dict(name="Debt Management",     score=round(dm  * 100, 1), weight=10, grade=grade(dm),
             insight=(f"Debt-to-income {dti*100:.0f}% — {'healthy' if dti < 0.35 else 'high'}."
                      if emi_amt else "No active EMI detected."),
             improvement="Keep EMI payments below 35% of monthly income."),
    ]


def loan_eligibility(score: int, avg_monthly_income: float) -> list:
    inc = avg_monthly_income
    if score >= 750:
        return [
            dict(product="Home Loan",          eligible=True,  max_amount=f"₹{int(inc*60):,}",  rate="8.5%–9.5%"),
            dict(product="Personal Loan",       eligible=True,  max_amount=f"₹{int(inc*24):,}",  rate="10.5%–13%"),
            dict(product="Car Loan",            eligible=True,  max_amount=f"₹{int(inc*36):,}",  rate="8.7%–10%"),
            dict(product="MUDRA Tarun (₹10L)",  eligible=True,  max_amount="₹10,00,000",          rate="9%–12%"),
        ]
    if score >= 650:
        return [
            dict(product="Personal Loan",       eligible=True,  max_amount=f"₹{int(inc*18):,}",  rate="13%–16%"),
            dict(product="MUDRA Kishore (₹5L)", eligible=True,  max_amount="₹5,00,000",           rate="12%–15%"),
            dict(product="Two-Wheeler Loan",    eligible=True,  max_amount=f"₹{int(inc*12):,}",  rate="11%–14%"),
            dict(product="Home Loan",           eligible=False, max_amount="—",                   rate="—"),
        ]
    if score >= 550:
        return [
            dict(product="MUDRA Shishu (₹50K)", eligible=True,  max_amount="₹50,000",             rate="15%–18%"),
            dict(product="Gold Loan",            eligible=True,  max_amount="Based on gold value", rate="12%–15%"),
            dict(product="Personal Loan",        eligible=False, max_amount="—",                   rate="—"),
        ]
    if score >= 450:
        return [
            dict(product="SHG Microfinance",     eligible=True,  max_amount="₹25,000",             rate="18%–24%"),
            dict(product="Secured Loan",          eligible=True,  max_amount="Based on asset",      rate="16%–20%"),
            dict(product="MUDRA Loan",            eligible=False, max_amount="—",                   rate="—"),
        ]
    return [
        dict(product="Credit Builder Program",   eligible=True,  max_amount="₹5,000–₹10,000",      rate="24%+"),
        dict(product="Standard Loans",           eligible=False, max_amount="Not eligible yet",     rate="—"),
    ]
