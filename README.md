## 🧠 Credit Risk Probability Model for Alternative Data

Welcome to the Week 5 project of the **10 Academy Artificial Intelligence Mastery Program**! In this project, we are tasked with building, deploying, and automating a **credit scoring model** using customer transaction data from a partner eCommerce platform. This model will help our client, **Bati Bank**, evaluate customer creditworthiness and enable **buy-now-pay-later (BNPL)** services.

---

### 📈 Business Need

Bati Bank is collaborating with a fast-growing eCommerce provider to offer credit services. The job is to build a machine learning pipeline that can:

1. Define a proxy credit risk variable (no default label exists).
2. Engineer meaningful features from raw behavioral data.
3. Train a model to predict credit risk.
4. Derive credit scores from predicted risk.
5. Suggest optimal loan amounts and durations.
6. Deploy the model as a scalable API with CI/CD and MLOps tools.

---

## 📂 Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml         # CI/CD workflow
├── data/
│   ├── raw/                         # Raw data
│   └── processed/                   # Cleaned & transformed data
├── notebooks/
│   └── 1.0-eda.ipynb                # Data exploration & insights
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Feature engineering logic
│   ├── train.py                    # Model training
│   ├── predict.py                  # Inference logic
│   └── api/
│       ├── main.py                 # FastAPI application
│       └── pydantic_models.py      # Request/response validation
├── tests/
│   └── test_data_processing.py     # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ✅ Tasks

### 🧠 Task 1: Credit Scoring Business Understanding

In real-world financial systems, interpretability and regulatory compliance are just as important as accuracy. Here’s what we’ve understood so far:

#### 📘 Why Basel II Emphasizes Model Transparency

The **Basel II Accord** requires banks to manage and justify credit risk assessments using internal models that are explainable and auditable. This means we must **build interpretable models** that regulators and auditors can understand, trust, and verify.

#### ⚠️ Why We Create a Proxy Variable

Our data doesn’t include a `default` label, so we define a **proxy credit risk variable** using customer behavior (Recency, Frequency, Monetary). This proxy lets us predict risk even without direct labels.

But it's important to note:

* The proxy might not reflect actual defaults.
* Mislabeling could hurt financial performance or fairness.
* We’ll need careful documentation, validation, and performance tracking to reduce risk.

#### ⚖️ Trade-offs: Interpretable vs. Complex Models

| Factor              | Logistic Regression with WoE  | Gradient Boosting Models         |
| ------------------- | ----------------------------- | -------------------------------- |
| Interpretability    | Very high                     |   Low                            |
| Performance         | Lower but stable              |   Higher accuracy                |
| Regulation-friendly | Easy to justify               |   Needs heavy explanation        |
| Use case            | First model in regulated bank | Complex relationships and tuning |

---

### 📊 Task 2: Exploratory Data Analysis (EDA)

* Explore structure, distribution, and key metrics in `1.0-eda.ipynb`
* Identify trends, missing data, and potential outliers
* Summary: Top 3–5 insights will guide our feature engineering.

### 🛠 Task 3: Feature Engineering

* Create aggregate features (total, mean, std. of transaction amounts)
* Extract datetime features (hour, day, month, year)
* Encode categorical features
* Handle missing data
* Normalize and build pipelines using `scikit-learn`

### 🎯 Task 4: Proxy Target Variable Engineering

* Calculate **RFM (Recency, Frequency, Monetary)** metrics
* Cluster customers with **KMeans** (3 segments)
* Label least active group as `is_high_risk = 1`
* Merge this label into the processed training data

### 🤖 Task 5: Model Training and Tracking

* Try 2+ models (e.g., Logistic Regression, Random Forest, XGBoost)
* Use MLflow for experiment tracking & model registry
* Evaluate using Accuracy, Precision, Recall, F1 Score, and ROC-AUC
* Write unit tests and register the best model

### 🚀 Task 6: Model Deployment & CI/CD

* Serve model with **FastAPI** and `uvicorn`
* Build Docker container and `docker-compose` for easy launch
* Add CI/CD pipeline with **GitHub Actions**:

  * Run `flake8` for linting
  * Run `pytest` for testing
* Pipeline blocks if tests or style checks fail

---

## 📚 Learning Outcomes

* 💡 Business reasoning with credit risk
* 🧼 Feature engineering & EDA
* ⚙️ Machine learning pipeline development
* 📦 Deployment with FastAPI, Docker, GitHub Actions
* 📊 Experiment tracking with MLflow
* 🔍 CI/CD, unit testing, logging, and more!