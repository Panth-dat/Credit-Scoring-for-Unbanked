"""
Mock Transaction Generator
Generates 18 months of realistic Indian bank transactions for any account number.
Deterministic: same account number always returns same transactions.
"""

import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict
import math

# ─── Category Definitions ────────────────────────────────────────────────────

CATEGORIES = {
    "BILL_PAYMENT": {
        "label": "Bill Payments",
        "emoji": "🔌",
        "merchants": ["BESCOM", "TATA POWER", "BSNL BROADBAND", "JIO FIBER",
                      "AIRTEL POSTPAID", "PIPED GAS MAHANAGAR", "WATER BOARD",
                      "BBMP PROPERTY TAX", "LIC PREMIUM"],
        "type": "debit",
        "frequency": "monthly",
        "amount_range": (500, 4500),
        "regularity": 0.90,   # how often they pay on time
    },
    "RECHARGE": {
        "label": "Mobile & DTH Recharge",
        "emoji": "📱",
        "merchants": ["JIO RECHARGE", "AIRTEL RECHARGE", "BSNL RECHARGE",
                      "VI RECHARGE", "TATA PLAY DTH", "DISH TV RECHARGE"],
        "type": "debit",
        "frequency": "monthly",
        "amount_range": (199, 999),
        "regularity": 0.95,
    },
    "GROCERY": {
        "label": "Groceries & Daily Needs",
        "emoji": "🛒",
        "merchants": ["DMART", "BIGBASKET", "BLINKIT", "ZEPTO", "RELIANCE FRESH",
                      "MORE SUPERMARKET", "SPENCER'S", "LOCAL KIRANA",
                      "SWIGGY INSTAMART", "AMAZON FRESH"],
        "type": "debit",
        "frequency": "weekly",
        "amount_range": (300, 3500),
        "regularity": 1.0,
    },
    "FOOD_DINING": {
        "label": "Food & Dining",
        "emoji": "🍔",
        "merchants": ["SWIGGY", "ZOMATO", "DOMINOS PIZZA", "MCDONALD'S",
                      "CAFE COFFEE DAY", "STARBUCKS", "LOCAL RESTAURANT",
                      "BARBEQUE NATION", "BOX8"],
        "type": "debit",
        "frequency": "weekly",
        "amount_range": (150, 2000),
        "regularity": 0.85,
    },
    "TRANSPORT": {
        "label": "Transport & Fuel",
        "emoji": "🚗",
        "merchants": ["OLAS CABS", "UBER INDIA", "RAPIDO", "INDIAN OIL",
                      "BHARAT PETROLEUM", "HP PETROL PUMP", "METRO CARD RECHARGE",
                      "BMTC BUS PASS", "FASTAG RECHARGE"],
        "type": "debit",
        "frequency": "weekly",
        "amount_range": (100, 4000),
        "regularity": 0.90,
    },
    "SHOPPING": {
        "label": "Shopping & Lifestyle",
        "emoji": "🛍️",
        "merchants": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "NYKAA",
                      "MEESHO", "H&M INDIA", "ZARA INDIA", "CROMA",
                      "RELIANCE DIGITAL"],
        "type": "debit",
        "frequency": "biweekly",
        "amount_range": (299, 12000),
        "regularity": 0.75,
    },
    "EMI_LOAN": {
        "label": "EMI & Loan Repayments",
        "emoji": "🏦",
        "merchants": ["HDFC HOME LOAN EMI", "SBI PERSONAL LOAN",
                      "BAJAJ FINSERV EMI", "ICICI AUTO LOAN",
                      "KOTAK LOAN EMI", "TATA CAPITAL EMI"],
        "type": "debit",
        "frequency": "monthly",
        "amount_range": (2000, 25000),
        "regularity": 0.88,
    },
    "SIP": {
        "label": "SIP & Mutual Funds",
        "emoji": "📈",
        "merchants": ["ZERODHA COIN SIP", "GROWW MF SIP", "PAYTM MONEY SIP",
                      "HDFC MF SIP", "SBI MF SIP", "AXIS MF SIP",
                      "MIRAE ASSET SIP", "NIPPON INDIA SIP"],
        "type": "debit",
        "frequency": "monthly",
        "amount_range": (500, 15000),
        "regularity": 0.95,
    },
    "STOCKS": {
        "label": "Stocks & Trading",
        "emoji": "📊",
        "merchants": ["ZERODHA BROKING", "UPSTOX", "ANGEL ONE",
                      "ICICI DIRECT", "HDFC SECURITIES", "5PAISA"],
        "type": "debit",
        "frequency": "irregular",
        "amount_range": (1000, 50000),
        "regularity": 0.60,
    },
    "INSURANCE": {
        "label": "Insurance Premiums",
        "emoji": "🛡️",
        "merchants": ["LIC OF INDIA", "HDFC LIFE", "ICICI PRUDENTIAL",
                      "MAX LIFE INSURANCE", "STAR HEALTH", "NIVA BUPA",
                      "BAJAJ ALLIANZ", "NEW INDIA ASSURANCE"],
        "type": "debit",
        "frequency": "monthly",
        "amount_range": (500, 8000),
        "regularity": 0.92,
    },
    "HEALTHCARE": {
        "label": "Healthcare & Pharmacy",
        "emoji": "🏥",
        "merchants": ["APOLLO PHARMACY", "MEDPLUS", "NETMEDS", "1MG",
                      "PRACTO CONSULT", "MANIPAL HOSPITAL", "NARAYANA HEALTH"],
        "type": "debit",
        "frequency": "irregular",
        "amount_range": (200, 8000),
        "regularity": 0.70,
    },
    "EDUCATION": {
        "label": "Education & Courses",
        "emoji": "📚",
        "merchants": ["BYJUS", "UNACADEMY", "COURSERA", "UDEMY",
                      "VEDANTU", "SCHOOL FEES", "COLLEGE FEES", "UPGRAD"],
        "type": "debit",
        "frequency": "monthly",
        "amount_range": (500, 20000),
        "regularity": 0.90,
    },
    "ENTERTAINMENT": {
        "label": "Entertainment & OTT",
        "emoji": "🎬",
        "merchants": ["NETFLIX", "AMAZON PRIME", "HOTSTAR DISNEY+",
                      "SONY LIV", "ZEE5", "BOOKMYSHOW", "PVR CINEMAS",
                      "SPOTIFY", "YOUTUBE PREMIUM"],
        "type": "debit",
        "frequency": "monthly",
        "amount_range": (99, 1500),
        "regularity": 0.88,
    },
    "UPI_TRANSFER": {
        "label": "UPI & Peer Transfers",
        "emoji": "💸",
        "merchants": ["UPI/", "PHONEPE/", "GPAY/", "PAYTM/"],  # prefix
        "type": "both",
        "frequency": "weekly",
        "amount_range": (100, 10000),
        "regularity": 1.0,
    },
    "SALARY": {
        "label": "Salary & Income",
        "emoji": "💰",
        "merchants": ["SALARY CREDIT", "NEFT CR-EMPLOYER", "PAYROLL CREDIT",
                      "STIPEND CREDIT", "FREELANCE PAYMENT NEFT"],
        "type": "credit",
        "frequency": "monthly",
        "amount_range": (15000, 120000),
        "regularity": 0.97,
    },
    "ATM_WITHDRAWAL": {
        "label": "ATM Withdrawals",
        "emoji": "🏧",
        "merchants": ["ATM WDL", "CASH WITHDRAWAL", "ATM CASH"],
        "type": "debit",
        "frequency": "biweekly",
        "amount_range": (1000, 10000),
        "regularity": 0.80,
    },
    "RENT": {
        "label": "Rent & Housing",
        "emoji": "🏠",
        "merchants": ["RENT PAYMENT NEFT", "HOUSE RENT UPI",
                      "LANDLORD TRANSFER", "PG RENT PAYMENT"],
        "type": "debit",
        "frequency": "monthly",
        "amount_range": (5000, 35000),
        "regularity": 0.95,
    },
    "TAX": {
        "label": "Taxes & Government",
        "emoji": "🏛️",
        "merchants": ["INCOME TAX PAYMENT", "GST PAYMENT", "TDS PAYMENT",
                      "ADVANCE TAX NSDL", "PROPERTY TAX"],
        "type": "debit",
        "frequency": "quarterly",
        "amount_range": (1000, 50000),
        "regularity": 0.85,
    },
    "CASHBACK_REFUND": {
        "label": "Cashback & Refunds",
        "emoji": "🎁",
        "merchants": ["CASHBACK CREDIT", "REFUND FROM AMAZON",
                      "FLIPKART REFUND", "ZOMATO REFUND", "GPAY CASHBACK"],
        "type": "credit",
        "frequency": "irregular",
        "amount_range": (10, 2000),
        "regularity": 0.50,
    },
    "INTEREST_DIVIDEND": {
        "label": "Interest & Dividends",
        "emoji": "💹",
        "merchants": ["SAVINGS INT CREDIT", "FD INTEREST CREDIT",
                      "DIVIDEND CREDIT", "RD MATURITY CREDIT"],
        "type": "credit",
        "frequency": "quarterly",
        "amount_range": (200, 15000),
        "regularity": 0.90,
    },
}

# ─── Profile Archetypes ───────────────────────────────────────────────────────

PROFILES = {
    "young_professional": {
        "salary_range": (35000, 90000),
        "active_categories": ["SALARY", "RENT", "GROCERY", "FOOD_DINING",
                               "TRANSPORT", "SHOPPING", "ENTERTAINMENT",
                               "RECHARGE", "BILL_PAYMENT", "SIP",
                               "UPI_TRANSFER", "CASHBACK_REFUND"],
        "score_band": (620, 780),
        "sip_probability": 0.70,
        "emi_probability": 0.30,
    },
    "family_earner": {
        "salary_range": (25000, 60000),
        "active_categories": ["SALARY", "RENT", "GROCERY", "BILL_PAYMENT",
                               "RECHARGE", "HEALTHCARE", "EDUCATION",
                               "INSURANCE", "EMI_LOAN", "TRANSPORT",
                               "UPI_TRANSFER", "ATM_WITHDRAWAL"],
        "score_band": (540, 700),
        "sip_probability": 0.40,
        "emi_probability": 0.65,
    },
    "investor": {
        "salary_range": (70000, 200000),
        "active_categories": ["SALARY", "GROCERY", "BILL_PAYMENT",
                               "SIP", "STOCKS", "INSURANCE", "TAX",
                               "SHOPPING", "ENTERTAINMENT", "INTEREST_DIVIDEND",
                               "UPI_TRANSFER", "RECHARGE"],
        "score_band": (720, 870),
        "sip_probability": 0.95,
        "emi_probability": 0.20,
    },
    "struggling": {
        "salary_range": (12000, 28000),
        "active_categories": ["SALARY", "BILL_PAYMENT", "GROCERY",
                               "RECHARGE", "ATM_WITHDRAWAL", "EMI_LOAN",
                               "TRANSPORT", "UPI_TRANSFER", "HEALTHCARE"],
        "score_band": (310, 490),
        "sip_probability": 0.05,
        "emi_probability": 0.80,
    },
    "gig_worker": {
        "salary_range": (18000, 50000),
        "active_categories": ["SALARY", "FOOD_DINING", "TRANSPORT",
                               "RECHARGE", "GROCERY", "ATM_WITHDRAWAL",
                               "SHOPPING", "ENTERTAINMENT", "UPI_TRANSFER",
                               "BILL_PAYMENT"],
        "score_band": (430, 610),
        "sip_probability": 0.25,
        "emi_probability": 0.40,
    },
}

# ─── Core Generator ───────────────────────────────────────────────────────────

def _seed_from_account(account_number: str) -> int:
    """Convert account number to deterministic integer seed."""
    return int(hashlib.md5(account_number.encode()).hexdigest(), 16) % (2**31)

def _pick_profile(rng: random.Random) -> dict:
    profiles = list(PROFILES.values())
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    return rng.choices(profiles, weights=weights, k=1)[0]

def _generate_dates(rng: random.Random, frequency: str, months: int = 18) -> List[datetime]:
    """Generate transaction dates for a given frequency over N months."""
    end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=months * 30)
    dates = []

    if frequency == "monthly":
        current = start.replace(day=rng.randint(1, 5))
        while current <= end:
            # Add 0-3 day jitter
            jittered = current + timedelta(days=rng.randint(-2, 3))
            if start <= jittered <= end:
                dates.append(jittered)
            current = (current.replace(day=1) + timedelta(days=32)).replace(day=current.day)

    elif frequency == "weekly":
        current = start
        while current <= end:
            if rng.random() < 0.85:
                jittered = current + timedelta(days=rng.randint(0, 2))
                if jittered <= end:
                    dates.append(jittered)
            current += timedelta(days=7)

    elif frequency == "biweekly":
        current = start
        while current <= end:
            if rng.random() < 0.80:
                dates.append(current)
            current += timedelta(days=14)

    elif frequency == "quarterly":
        current = start
        while current <= end:
            if rng.random() < 0.88:
                dates.append(current)
            current += timedelta(days=91)

    elif frequency == "irregular":
        num_txns = rng.randint(3, 12)
        span = (end - start).days
        for _ in range(num_txns):
            dates.append(start + timedelta(days=rng.randint(0, span)))

    return sorted(dates)


def generate_transactions(account_number: str, months: int = 18) -> Dict:
    """
    Main entry point. Returns full transaction history + metadata.
    Deterministic: same account → same transactions.
    """
    rng = random.Random(_seed_from_account(account_number))
    profile = _pick_profile(rng)

    salary = rng.randint(*profile["salary_range"])
    transactions = []
    txn_id = 1

    for cat_key, cat_data in CATEGORIES.items():
        if cat_key not in profile["active_categories"]:
            continue

        # Special handling for optional categories
        if cat_key == "SIP" and rng.random() > profile["sip_probability"]:
            continue
        if cat_key == "EMI_LOAN" and rng.random() > profile["emi_probability"]:
            continue
        if cat_key == "STOCKS" and "STOCKS" not in profile["active_categories"]:
            continue

        dates = _generate_dates(rng, cat_data["frequency"], months)

        for date in dates:
            # Skip some transactions based on regularity
            if rng.random() > cat_data["regularity"]:
                continue

            # Amount
            lo, hi = cat_data["amount_range"]
            if cat_key == "SALARY":
                # Salary is consistent with small variation
                amount = salary + rng.randint(-2000, 2000)
            else:
                amount = round(rng.uniform(lo, hi), 2)

            # Merchant
            merchant_list = cat_data["merchants"]
            if cat_key == "UPI_TRANSFER":
                prefix = rng.choice(merchant_list)
                names = ["Rahul S", "Priya K", "Amit V", "Sneha M", "Ravi T",
                         "Mom", "Dad", "Landlord", "Office Colleague", "Friend"]
                merchant = prefix + rng.choice(names).upper().replace(" ", "_")
            else:
                merchant = rng.choice(merchant_list)

            # Transaction type
            if cat_key == "UPI_TRANSFER":
                txn_type = rng.choice(["debit", "credit"])
            else:
                txn_type = cat_data["type"]

            # Add some late payment markers for bills/EMI
            is_late = False
            if cat_key in ["BILL_PAYMENT", "EMI_LOAN", "RENT"]:
                is_late = rng.random() < (1 - cat_data["regularity"])

            transactions.append({
                "txn_id": f"TXN{str(txn_id).zfill(6)}",
                "date": date.strftime("%Y-%m-%d"),
                "month": date.strftime("%B %Y"),
                "merchant": merchant,
                "category": cat_key,
                "category_label": cat_data["label"],
                "category_emoji": cat_data["emoji"],
                "amount": amount,
                "type": txn_type,
                "is_late_payment": is_late,
                "balance_after": None,  # computed below
            })
            txn_id += 1

    # Sort by date
    transactions.sort(key=lambda x: x["date"])

    # Compute running balance
    balance = rng.randint(5000, 50000)  # opening balance
    for txn in transactions:
        if txn["type"] == "credit":
            balance += txn["amount"]
        else:
            balance = max(0, balance - txn["amount"])
        txn["balance_after"] = round(balance, 2)

    # ─── Category Summary ─────────────────────────────────────────────────────
    category_summary = {}
    for txn in transactions:
        key = txn["category"]
        label = txn["category_label"]
        emoji = txn["category_emoji"]
        if key not in category_summary:
            category_summary[key] = {
                "category": key,
                "label": label,
                "emoji": emoji,
                "total_debit": 0,
                "total_credit": 0,
                "transaction_count": 0,
                "avg_amount": 0,
                "late_payments": 0,
                "monthly_avg": 0,
            }
        s = category_summary[key]
        s["transaction_count"] += 1
        if txn["type"] == "debit":
            s["total_debit"] += txn["amount"]
        else:
            s["total_credit"] += txn["amount"]
        if txn["is_late_payment"]:
            s["late_payments"] += 1

    for key, s in category_summary.items():
        total = s["total_debit"] + s["total_credit"]
        s["avg_amount"] = round(total / s["transaction_count"], 2) if s["transaction_count"] else 0
        s["monthly_avg"] = round(total / months, 2)
        s["total_debit"] = round(s["total_debit"], 2)
        s["total_credit"] = round(s["total_credit"], 2)

    # ─── Monthly Summary ──────────────────────────────────────────────────────
    monthly_summary = {}
    for txn in transactions:
        m = txn["month"]
        if m not in monthly_summary:
            monthly_summary[m] = {"month": m, "total_debit": 0, "total_credit": 0, "txn_count": 0}
        monthly_summary[m]["txn_count"] += 1
        if txn["type"] == "debit":
            monthly_summary[m]["total_debit"] += txn["amount"]
        else:
            monthly_summary[m]["total_credit"] += txn["amount"]

    for m in monthly_summary:
        monthly_summary[m]["total_debit"] = round(monthly_summary[m]["total_debit"], 2)
        monthly_summary[m]["total_credit"] = round(monthly_summary[m]["total_credit"], 2)
        monthly_summary[m]["net"] = round(
            monthly_summary[m]["total_credit"] - monthly_summary[m]["total_debit"], 2
        )

    # ─── Credit Signal Features (for ML model) ───────────────────────────────
    all_debits = [t["amount"] for t in transactions if t["type"] == "debit"]
    all_credits = [t["amount"] for t in transactions if t["type"] == "credit"]
    salary_txns = [t["amount"] for t in transactions if t["category"] == "SALARY"]
    sip_txns = [t["amount"] for t in transactions if t["category"] == "SIP"]
    emi_txns = [t["amount"] for t in transactions if t["category"] == "EMI_LOAN"]
    bill_txns = [t for t in transactions if t["category"] == "BILL_PAYMENT"]
    late_txns = [t for t in transactions if t["is_late_payment"]]

    total_income = sum(all_credits)
    total_expense = sum(all_debits)
    avg_monthly_income = total_income / months
    avg_monthly_expense = total_expense / months

    features = {
        # Income signals
        "avg_monthly_income": round(avg_monthly_income, 2),
        "income_consistency_score": round(
            1 - (max(salary_txns) - min(salary_txns)) / max(max(salary_txns), 1)
            if len(salary_txns) > 1 else 0.9, 3),
        "num_income_sources": len(set(
            t["category"] for t in transactions if t["type"] == "credit"
        )),

        # Savings signals
        "avg_monthly_savings": round(avg_monthly_income - avg_monthly_expense, 2),
        "savings_rate": round(
            max(0, (avg_monthly_income - avg_monthly_expense) / avg_monthly_income)
            if avg_monthly_income > 0 else 0, 3),

        # Investment signals
        "has_sip": len(sip_txns) > 0,
        "sip_monthly_avg": round(sum(sip_txns) / months, 2) if sip_txns else 0,
        "has_stocks": any(t["category"] == "STOCKS" for t in transactions),
        "investment_rate": round(
            (sum(sip_txns)) / total_income if total_income > 0 else 0, 3),

        # Debt signals
        "has_emi": len(emi_txns) > 0,
        "emi_monthly_avg": round(sum(emi_txns) / months, 2) if emi_txns else 0,
        "debt_to_income_ratio": round(
            (sum(emi_txns) / months) / avg_monthly_income
            if avg_monthly_income > 0 else 0, 3),

        # Payment behavior
        "bill_payment_count": len(bill_txns),
        "late_payment_count": len(late_txns),
        "late_payment_rate": round(
            len(late_txns) / len(bill_txns) if bill_txns else 0, 3),
        "on_time_payment_rate": round(
            1 - (len(late_txns) / len(bill_txns)) if bill_txns else 1.0, 3),

        # Spending patterns
        "grocery_monthly_avg": round(
            sum(t["amount"] for t in transactions if t["category"] == "GROCERY") / months, 2),
        "entertainment_monthly_avg": round(
            sum(t["amount"] for t in transactions if t["category"] == "ENTERTAINMENT") / months, 2),
        "healthcare_spend": round(
            sum(t["amount"] for t in transactions if t["category"] == "HEALTHCARE"), 2),

        # Stability signals
        "total_transactions": len(transactions),
        "avg_txns_per_month": round(len(transactions) / months, 1),
        "account_active_months": months,
        "min_monthly_balance_proxy": round(min(
            t["balance_after"] for t in transactions if t["balance_after"] is not None
        ), 2) if transactions else 0,

        # Digital behavior
        "upi_transaction_count": sum(
            1 for t in transactions if t["category"] == "UPI_TRANSFER"),
        "has_insurance": any(t["category"] == "INSURANCE" for t in transactions),
        "has_education_spend": any(t["category"] == "EDUCATION" for t in transactions),
    }

    return {
        "account_number": account_number,
        "account_holder": _generate_name(rng),
        "profile_type": [k for k, v in PROFILES.items() if v == profile][0],
        "analysis_period_months": months,
        "analysis_from": (datetime.now() - timedelta(days=months * 30)).strftime("%d %b %Y"),
        "analysis_to": datetime.now().strftime("%d %b %Y"),
        "total_transactions": len(transactions),
        "transactions": transactions,
        "category_summary": list(category_summary.values()),
        "monthly_summary": list(monthly_summary.values()),
        "credit_features": features,
    }


def _generate_name(rng: random.Random) -> str:
    first = ["Amit", "Priya", "Rahul", "Sneha", "Vijay", "Anita", "Suresh",
             "Kavitha", "Ravi", "Deepa", "Arun", "Meena", "Kiran", "Pooja",
             "Manoj", "Lakshmi", "Sanjay", "Nisha", "Ajay", "Rekha"]
    last = ["Sharma", "Patel", "Kumar", "Singh", "Reddy", "Nair", "Joshi",
            "Mehta", "Gupta", "Iyer", "Pillai", "Rao", "Verma", "Das", "Shah"]
    return f"{rng.choice(first)} {rng.choice(last)}"


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    result = generate_transactions("ACC1234567890")
    print(f"Account: {result['account_holder']}")
    print(f"Profile: {result['profile_type']}")
    print(f"Total Transactions: {result['total_transactions']}")
    print(f"\nCategory Summary:")
    for cat in sorted(result['category_summary'], key=lambda x: x['total_debit'] + x['total_credit'], reverse=True):
        total = cat['total_debit'] + cat['total_credit']
        print(f"  {cat['emoji']} {cat['label']:<30} ₹{total:>10,.0f}  ({cat['transaction_count']} txns)")
    print(f"\nCredit Features:")
    for k, v in result['credit_features'].items():
        print(f"  {k:<40} {v}")
