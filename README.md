# HDFC Bank - Loan Approval Prediction System

A supervised machine learning classification project and web frontend to predict whether a loan application should be **Approved** or **Rejected** based on applicant income, loan amount, credit score (CIBIL), employment status, and asset indicators.

---

## Project Structure

```
FINAL/
├── index.html                      # Web Frontend (UI)
├── style.css                       # Styling for Frontend
├── script.js                       # Frontend Machine Learning Inference Engine
├── loan_approval_dataset.csv       # HDFC Bank Loan Dataset (4,269 records)
├── Loan_Approval_Prediction.ipynb  # College Lab Jupyter Notebook (pre-executed)
├── main.py                         # Python ML Pipeline Script
├── requirements.txt                # Python Dependencies
├── README.md                       # Project Documentation
├── src/                            # Modular Python Source Code
│   ├── data_loader.py              # Data cleaning and loading
│   ├── feature_engineering.py      # Banking financial ratios
│   ├── model_trainer.py            # Model training & cross-validation
│   ├── evaluator.py                # Performance metrics & reports
│   └── predictor.py                # Python inference engine
```

---

## How to Run Locally

### 1. Run the Web Frontend
Simply double-click **`index.html`** or right-click and open it in any browser (Chrome, Edge, Firefox).
* No local server or installation needed.
* You can test with custom inputs or use the sample profile buttons (**Good Profile**, **Risky Profile**, **Self-Employed**).

### 2. Run the Python Backend Pipeline
To train the models and test the Python pipeline:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main pipeline
python main.py
```

### 3. Open the College Notebook
Open **`Loan_Approval_Prediction.ipynb`** in VS Code or Jupyter Notebook.
* Contains step-by-step data cleaning, exploratory data analysis, model comparison (Logistic Regression, Decision Tree, Random Forest), and feature importance plots.
* All outputs are already pre-computed and rendered for easy presentation.

---

## Model Highlights

- **Dataset Size:** 4,269 historical loan applications
- **Best Model:** Random Forest Classifier (**97.9% test accuracy**)
- **Key Factor:** **CIBIL credit score** is the primary driver (accounts for >85% of decisions), followed by total asset coverage and loan-to-income ratio.
