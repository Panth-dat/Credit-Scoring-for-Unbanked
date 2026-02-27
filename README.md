# CreditIQ — Transaction-Based Credit Scoring Platform

> Predicts CIBIL credit scores (300–900) from 18 months of bank transaction history using Machine Learning.

---

## Quick Start (Windows)

1. Make sure **Python 3.10+** is installed → https://python.org
2. Double-click **`start.bat`**
3. Browser opens automatically at **http://localhost:8000**

That's it. `start.bat` handles all dependency installation and model training automatically.

---

## Demo Accounts (no setup needed)

| Account Number    | Profile           | Expected Score |
|-------------------|-------------------|---------------|
| DEMO001INVESTOR   | Investor          | High (750+)   |
| DEMO002YOUNG      | Young Professional| Good (650-749)|
| DEMO003FAMILY     | Family Earner     | Fair (550-649)|
| DEMO004GIG        | Gig Worker        | Poor (450-549)|
| DEMO005STRUGGLE   | Struggling        | Very Poor (<450)|

---

## How the Model Works

The system replicates the exact notebook pipeline:

### Step 1 — Transaction Categorization
- **Model**: LogisticRegression with TF-IDF vectorizer
- **Input**: Transaction description text (e.g. `UPI/DR/pay@swiggy/SWIGGY FOOD/123456`)
- **Output**: 20 category labels (SALARY, SIP, BILL_PAYMENT, GROCERY, etc.)
- **Accuracy**: ~99.2%

### Step 2 — Credit Score Prediction
- **Model**: RandomForestClassifier (300 trees, max_depth=10)
- **Features** (exactly 7, as in notebook):
  ```
  amount_sum         — total transaction volume
  amount_mean        — average transaction size
  amount_std         — transaction volatility
  type_sum           — count of credit transactions
  is_late_payment_sum— number of late payments
  balance_after_mean — average running balance
  balance_after_min  — lowest balance reached
  ```
- **Risk label**: 1 if (late_payments > 3) OR (total_amount < 5000) OR (min_balance < 1000)
- **CIBIL Score**: `900 − (risk_probability × 600)` → range 300–900

---

## Project Structure

```
credit-score-platform/
├── start.bat                  ← Double-click to run everything
├── README.md
└── backend/
    ├── main.py                ← FastAPI app (serves frontend + API)
    ├── train_model.py         ← Trains and saves models
    ├── predict.py             ← Prediction logic (exact notebook replica)
    ├── transaction_generator.py ← Mock bank API (deterministic)
    ├── requirements.txt
    └── models/
        ├── category_model.pkl
        ├── tfidf_vectorizer.pkl
        ├── label_encoder.pkl
        └── risk_model.pkl
frontend/
    └── index.html             ← Full dashboard (served by FastAPI)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serves the dashboard UI |
| `POST` | `/api/score` | Full score analysis for an account |
| `GET`  | `/api/score/{account}` | Same via GET |
| `POST` | `/api/transactions` | Raw transaction data |
| `POST` | `/api/whatif` | Simulate score with modified habits |
| `GET`  | `/api/demo-accounts` | List of demo accounts |
| `POST` | `/api/categorize` | Classify a transaction description |
| `GET`  | `/docs` | Swagger interactive API docs |

### Example: Get a credit score
```bash
curl -X POST http://localhost:8000/api/score \
  -H "Content-Type: application/json" \
  -d '{"account_number": "DEMO001INVESTOR"}'
```

---

## Retrain Models

If you want to retrain from scratch:
```bash
cd backend
python train_model.py
```

---

## Requirements

- Python 3.10+
- No Node.js required (frontend is plain HTML served by FastAPI)
- Internet not required after first `pip install`
