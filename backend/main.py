from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from transaction_generator import generate_transactions
from predict import predict_cibil, score_components, loan_eligibility, predict_category

app = FastAPI(title="CreditIQ API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND = pathlib.Path(__file__).parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

_cache: dict = {}

# ── request models ───────────────────────────────────────────────────────────
class AccountReq(BaseModel):
    account_number: str = Field(..., min_length=3)
    months: Optional[int] = 18

class RawTxn(BaseModel):
    date: str
    description: str
    amount: float
    type: str           # "CREDIT" or "DEBIT"
    balance_after: Optional[float] = 0.0

class AnalyzeReq(BaseModel):
    transactions: List[RawTxn]
    account_holder: Optional[str] = "Account Holder"

class WhatIfReq(BaseModel):
    account_number: str
    on_time_rate: Optional[float] = None
    has_sip: Optional[bool] = None

# ── helpers ──────────────────────────────────────────────────────────────────
def _is_credit(t: dict) -> bool:
    return str(t.get("type", "")).lower() in ("credit", "cr")

def _is_late(t: dict) -> bool:
    lp = t.get("is_late_payment", False)
    if isinstance(lp, bool): return lp
    return str(lp).upper() in ("YES", "1", "TRUE")

def _to100(s300: int) -> int:
    return max(0, min(100, round((s300 - 300) / 6)))

def _interp(s: int) -> dict:
    if s >= 80: return dict(grade="Excellent", color="#15803d", bg="#dcfce7",
        risk_level="Very Low Risk",
        title="Exceptional Credit Profile",
        description="Your financial behaviour is outstanding. You are very likely to qualify for premium loan products at the lowest available interest rates.",
        next_step="Apply for a home loan or increase your credit limit — you have maximum negotiating power with lenders.")
    if s >= 65: return dict(grade="Good", color="#16a34a", bg="#d1fae5",
        risk_level="Low Risk",
        title="Good Credit Profile",
        description="Solid financial habits with minor areas to improve. Most lenders will view your profile favourably and approve most loan applications.",
        next_step="Pay all bills on time consistently for 6 months and reduce your DTI ratio to reach Excellent range.")
    if s >= 50: return dict(grade="Fair", color="#ca8a04", bg="#fef9c3",
        risk_level="Medium Risk",
        title="Fair Credit Profile",
        description="Your profile shows some financial stress signals. You may qualify for loans but will likely be offered higher interest rates.",
        next_step="Clear any pending late payments and start a ₹500/month SIP — both actions show significant results within 3 months.")
    if s >= 35: return dict(grade="Poor", color="#ea580c", bg="#ffedd5",
        risk_level="High Risk",
        title="Poor Credit Profile",
        description="Significant financial stress detected. Loan approval will be difficult with most mainstream lenders.",
        next_step="Build a 3-month emergency fund first. Then set up auto-debit for every recurring bill to eliminate late payments.")
    return dict(grade="Very Poor", color="#dc2626", bg="#fee2e2",
        risk_level="Very High Risk",
        title="Very Poor Credit Profile",
        description="High financial risk detected. Immediate corrective action is required to avoid further deterioration.",
        next_step="Stop taking new loans immediately. Consult a certified financial advisor and start with a secured credit product to rebuild history.")

LATE_KEYWORDS = ["LATE FEE","LATE CHARGE","PENALTY","OVERDUE","BOUNCE",
                  "DISHONOUR","INSUFFICIENT","RETURN CHARGE","ECS RETURN",
                  "NACH RETURN","PAYMENT FAILED","DELINQUENT","DEFAULT"]

def _full_analysis(txns: list, account_number: str, account_holder: str,
                   period_from: str, period_to: str) -> dict:
    months = 18

    raw      = predict_cibil(txns)
    comps    = score_components(txns)
    score    = _to100(raw["cibil_score"])
    interp   = _interp(score)

    credits_ = [t for t in txns if _is_credit(t)]
    debits   = [t for t in txns if not _is_credit(t)]
    late     = [t for t in txns if _is_late(t)]
    bills    = [t for t in txns if t.get("category","") in
                ("BILL_PAYMENT","EMI_LOAN","RENT","INSURANCE")]

    avg_inc  = sum(t["amount"] for t in credits_) / months if credits_ else 0
    avg_exp  = sum(t["amount"] for t in debits)   / months if debits   else 0
    sav_r    = max(0.0, (avg_inc - avg_exp) / avg_inc) if avg_inc else 0
    sip_amt  = sum(t["amount"] for t in debits if t.get("category") == "SIP")
    emi_amt  = sum(t["amount"] for t in debits if t.get("category") == "EMI_LOAN")

    # per-month buckets (keyed by YYYY-MM)
    monthly: dict = {}
    for t in txns:
        ym = str(t.get("date", ""))[:7]
        if not ym: continue
        if ym not in monthly:
            monthly[ym] = {"ym": ym, "income": 0.0, "expense": 0.0, "count": 0}
        if _is_credit(t): monthly[ym]["income"]  += t["amount"]
        else:              monthly[ym]["expense"] += t["amount"]
        monthly[ym]["count"] += 1
    for m in monthly.values():
        m["income"]  = round(m["income"],  2)
        m["expense"] = round(m["expense"], 2)
        m["net"]     = round(m["income"] - m["expense"], 2)
    monthly_list = sorted(monthly.values(), key=lambda x: x["ym"])

    # category buckets
    cats: dict = {}
    for t in txns:
        c = t.get("category", "OTHER") or "OTHER"
        if c not in cats:
            cats[c] = {"category": c, "label": c.replace("_", " ").title(),
                       "total_debit": 0.0, "total_credit": 0.0, "count": 0}
        cats[c]["count"] += 1
        if _is_credit(t): cats[c]["total_credit"] += t["amount"]
        else:              cats[c]["total_debit"]  += t["amount"]
    for c in cats.values():
        c["total_debit"]  = round(c["total_debit"],  2)
        c["total_credit"] = round(c["total_credit"], 2)

    # clean transactions
    clean = [{"date": t.get("date",""), "merchant": t.get("merchant", t.get("description","")),
              "category": t.get("category","OTHER"), "amount": round(float(t.get("amount",0)),2),
              "type": "CREDIT" if _is_credit(t) else "DEBIT",
              "balance_after": round(float(t.get("balance_after",0) or 0),2),
              "is_late": _is_late(t)} for t in txns]

    loans = loan_eligibility(raw["cibil_score"], avg_inc)

    return {
        "account_number":    account_number,
        "account_holder":    account_holder,
        "analysis_period":   f"{period_from} to {period_to}",
        "total_transactions": len(txns),
        "score":             score,
        "grade":             interp["grade"],
        "grade_color":       interp["color"],
        "grade_bg":          interp["bg"],
        "interpretation":    interp,
        "components":        comps,
        "loan_eligibility":  loans,
        "financial_summary": {
            "avg_monthly_income":  round(avg_inc, 2),
            "avg_monthly_expense": round(avg_exp, 2),
            "avg_monthly_savings": round(avg_inc - avg_exp, 2),
            "savings_rate_pct":    round(sav_r * 100, 1),
            "has_sip":             sip_amt > 0,
            "sip_monthly_avg":     round(sip_amt / months, 2),
            "has_emi":             emi_amt > 0,
            "emi_monthly_avg":     round(emi_amt / months, 2),
            "dti_pct":             round((emi_amt/months/avg_inc*100) if avg_inc else 0, 1),
            "on_time_pct":         round((1 - len(late)/max(len(bills),1))*100, 1),
            "late_count":          len(late),
            "total_debits":        round(sum(t["amount"] for t in debits), 2),
        },
        "category_summary":  sorted(cats.values(), key=lambda x: x["total_debit"], reverse=True),
        "monthly_summary":   monthly_list,
        "transactions":      clean,
    }

def _build_demo(account_number: str, months: int = 18) -> dict:
    key = f"{account_number}:{months}"
    if key not in _cache:
        _cache[key] = generate_transactions(account_number.strip().upper(), months)
    data = _cache[key]
    txns = data["transactions"]
    return _full_analysis(txns, data["account_number"], data["account_holder"],
                          data["analysis_from"], data["analysis_to"])

# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    idx = FRONTEND / "index.html"
    return FileResponse(str(idx)) if idx.exists() else {"status": "ok"}

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/api/score")
def score_demo(req: AccountReq):
    try: return _build_demo(req.account_number, req.months or 18)
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/score/{account_number}")
def score_demo_get(account_number: str, months: int = 18):
    try: return _build_demo(account_number, months)
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/analyze")
def analyze_csv(req: AnalyzeReq):
    """Analyze transactions from uploaded CSV."""
    try:
        if len(req.transactions) < 30:
            raise HTTPException(400, f"Only {len(req.transactions)} transactions found. Need at least 30.")
        txns = []
        dates = []
        for t in req.transactions:
            try:  cat = predict_category(t.description)
            except: cat = "OTHER"
            desc_up = t.description.upper()
            is_late = any(kw in desc_up for kw in LATE_KEYWORDS)
            txns.append({"date": t.date, "merchant": t.description, "category": cat,
                         "amount": abs(float(t.amount)), "type": t.type.lower(),
                         "is_late_payment": is_late, "balance_after": float(t.balance_after or 0)})
            if t.date: dates.append(t.date[:10])
        dates.sort()
        period_from = dates[0]  if dates else "—"
        period_to   = dates[-1] if dates else "—"
        return _full_analysis(txns, "UPLOAD", req.account_holder, period_from, period_to)
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/whatif")
def what_if(req: WhatIfReq):
    try:
        key  = f"{req.account_number}:18"
        data = _cache.get(key) or generate_transactions(req.account_number.strip().upper(), 18)
        txns = data["transactions"]
        base = predict_cibil(txns)
        import random; rng = random.Random(42)
        modified = []
        for t in txns:
            m = dict(t)
            if req.on_time_rate is not None and _is_late(t):
                m["is_late_payment"] = not (rng.random() < req.on_time_rate)
            modified.append(m)
        if req.has_sip:
            from datetime import datetime, timedelta
            for i in range(18):
                dt = (datetime(2025,6,30)-timedelta(days=i*30)).strftime("%Y-%m-%d")
                modified.append({"date":dt,"merchant":"HDFC MF SIP","category":"SIP",
                    "amount":3000,"type":"credit","is_late_payment":False,"balance_after":10000})
        proj = predict_cibil(modified)
        b100, p100 = _to100(base["cibil_score"]), _to100(proj["cibil_score"])
        delta = p100 - b100
        interp = _interp(p100)
        return {"original_score":b100,"projected_score":p100,"delta":delta,
                "projected_grade":interp["grade"],"projected_color":interp["color"],
                "message":f"Score could {'rise' if delta>=0 else 'fall'} by {abs(delta)} pts to {p100}/100 ({interp['grade']})."}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/demo-accounts")
def demos():
    return {"accounts":[
        {"account":"DEMO001INVESTOR","name":"Priya Nair",   "profile":"Investor",          "hint":"~94/100"},
        {"account":"DEMO002YOUNG",   "name":"Amit Sharma",  "profile":"Young Professional","hint":"~94/100"},
        {"account":"DEMO003FAMILY",  "name":"Rahul Verma",  "profile":"Family Earner",     "hint":"~100/100"},
        {"account":"DEMO004GIG",     "name":"Ravi Shankar", "profile":"Gig Worker",        "hint":"~76/100"},
        {"account":"DEMO005STRUGGLE","name":"Kavitha Reddy","profile":"Struggling",        "hint":"~60/100"},
    ]}
