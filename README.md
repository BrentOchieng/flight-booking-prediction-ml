#  British Airways Customer Booking Prediction

> **Forage Job Simulation | Machine Learning Classification Project**
> Predicting customer booking completion using Random Forest and XGBoost — helping British Airways prioritize high-conversion customers and optimize marketing spend.

---

##  Repository Info

| Field | Detail |
|---|---|
| **Short Description** | Machine learning pipeline to predict British Airways customer booking completion using Random Forest & XGBoost — built as part of the British Airways Data Science Virtual Job Simulation on Forage. |
| **Topics/Tags** | `machine-learning` `xgboost` `random-forest` `classification` `airline-analytics` `customer-behavior` `python` `sklearn` `data-science` `forage` `feature-engineering` `imbalanced-classification` |

---

##  Table of Contents

- [Business Problem](#-business-problem)
- [Dataset Overview](#-dataset-overview)
- [Project Workflow](#-project-workflow)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis-eda)
  - [2. Feature Engineering & Encoding](#2-feature-engineering--encoding)
  - [3. Train-Test Split & Scaling](#3-train-test-split--scaling)
  - [4. Model 1 — Random Forest Classifier](#4-model-1--random-forest-classifier)
  - [5. Feature Importance & Noise Reduction](#5-feature-importance--noise-reduction)
  - [6. Model 2 — XGBoost Classifier](#6-model-2--xgboost-classifier)
  - [7. ROC-AUC Comparison](#7-roc-auc-comparison)
- [Final Results](#-final-results)
- [Testing with New Data](#-testing-the-model-with-new-data)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Key Takeaways](#-key-takeaways)

---

##  Business Problem

Airlines invest heavily in attracting potential customers — yet a significant portion never complete their bookings. This churn represents lost revenue and wasted marketing spend.

**British Airways** needs to understand *which customers are most likely to complete a booking* based on behavioral signals captured during the booking journey.

A predictive model enables British Airways to:

-  **Target high-potential customers** with personalized interventions
-  **Optimize marketing budgets** by focusing on likely converters
-  **Improve conversion rates** through data-driven engagement
-  **Enhance customer experience** with relevant, timely offers

**Objective:** Build a binary classification model that predicts `booking_complete` (1 = completed, 0 = not completed) using historical customer booking data.

---

##  Dataset Overview

**Source:** `customer_booking.xlsx` — British Airways historical booking dataset

**Shape:** ~50,000 rows × 14 columns

| Column | Description |
|---|---|
| `num_passengers` | Number of passengers travelling |
| `sales_channel` | Channel used to make the booking (Mobile / Internet) |
| `trip_type` | Trip type: Round Trip, One Way, Circle Trip |
| `purchase_lead` | Days between booking date and travel date |
| `length_of_stay` | Number of days spent at destination |
| `flight_hour` | Hour of flight departure |
| `flight_day` | Day of the week of flight departure |
| `route` | Origin → Destination flight route |
| `booking_origin` | Country from which the booking was made |
| `wants_extra_baggage` | Whether the customer opted for extra baggage |
| `wants_preferred_seat` | Whether the customer selected a preferred seat |
| `wants_in_flight_meals` | Whether the customer ordered in-flight meals |
| `flight_duration` | Total flight duration in hours |
| `booking_complete` | **Target variable** — 1 if booking was completed, 0 otherwise |

**Data Quality Checks:**
-  No missing or null values found
-  No duplicate records
-  Class imbalance detected — `booking_complete = 1` is the minority class, addressed using `class_weight='balanced'` and `scale_pos_weight`

---

##  Project Workflow

### 1. Exploratory Data Analysis (EDA)

Initial data exploration was performed to understand distributions, detect outliers, and uncover patterns in booking behavior.

**Key EDA Steps:**
- Examined dataset shape, column types, and statistical summaries (`df.describe()`, `df.info()`)
- Checked for missing values and skewness using histograms
- Detected outliers via boxplots
- Visualized booking completion rates by:
  - **Sales channel** (Mobile vs Internet) — Internet users showed higher booking completion
  - **Trip type** (Round Trip, One Way, Circle Trip) — Round trips were the most common

```python
# Most popular booking channel by completion
sns.countplot(data=df, x='booking_complete', hue='sales_channel')

# Trip type preference
sns.countplot(data=df, x='booking_complete', hue='trip_type')
```

---

### 2. Feature Engineering & Encoding

A focused feature subset was selected for modelling based on relevance to customer behavior:

**Selected Features:**
`sales_channel`, `length_of_stay`, `flight_hour`, `flight_day`, `purchase_lead`, `trip_type`

**Encoding Strategy:**

| Feature | Encoding Method | Reason |
|---|---|---|
| `flight_day` | One-Hot Encoding (`pd.get_dummies`) | Nominal categorical — no ordinal relationship |
| `trip_type` | One-Hot Encoding (`pd.get_dummies`) | Nominal categorical |
| `sales_channel` | Binary Label Encoding (Mobile=0, Internet=1) | Binary categorical |

**Engineered Feature — Booking Window Ratio:**

A new interaction feature was created to capture *how far in advance a customer books relative to the length of their stay*:

```python
df1['booking_window_ratio'] = df1['purchase_lead'] / (df1['length_of_stay'] + 1)
```

> The intuition: a customer booking 60 days ahead for a 2-day trip behaves very differently from one booking 60 days ahead for a 3-week trip. This ratio captures that signal.

---

### 3. Train-Test Split & Scaling

- **Split Ratio:** 80% Training / 20% Testing
- **Stratification:** Applied to preserve the class distribution in both sets
- **Scaling:** `StandardScaler` fitted **only on training data**, then applied to both train and test sets to prevent data leakage

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

>  **Why scale before splitting is wrong:** Fitting the scaler on the full dataset would allow test data statistics to influence the scaler — a form of data leakage that inflates performance metrics.

---

### 4. Model 1 — Random Forest Classifier

```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',   # handles class imbalance
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
```

**Hyperparameters:**
- `n_estimators=200` — 200 decision trees averaged for robustness
- `class_weight='balanced'` — automatically adjusts weights inversely proportional to class frequencies
- `max_depth=10` — prevents overfitting by limiting tree depth

**Threshold Tuning:**

The default 0.5 probability threshold was adjusted to 0.4 to improve recall on the minority class (completed bookings):

```python
y_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
y_pred_adjusted = (y_probs > 0.4).astype(int)
```

**Initial Confusion Matrix Results:**

| Metric | Value |
|---|---|
| True Negatives (correctly predicted "Not Booked") | 5,284 |
| True Positives (correctly predicted "Booked") | 860 |
| False Negatives (missed actual bookings) | 636 |
| False Positives (wrongly predicted as booked) | 3,220 |

**Classification Report Highlights:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 — Not Booked | 89% | 62% | 73% |
| 1 — Booked | 21% | 57% | 31% |

---

### 5. Feature Importance & Noise Reduction

Feature importances from the initial Random Forest model revealed that **individual flight day dummies contributed very little** to model decisions — introducing noise without predictive value.

**Action Taken:**
- Dropped all `flight_day_*` dummy columns
- Added the engineered `booking_window_ratio` feature
- Retrained on the refined feature set (`X1`)

```python
X1 = X.drop(['flight_day_Mon', 'flight_day_Thu', 'flight_day_Wed',
              'flight_day_Tue', 'flight_day_Sat', 'flight_day_Sun'], axis=1)

X1 = pd.concat([X1, df1[['booking_window_ratio']]], axis=1)
```

**Top Predictive Features (Post-Refinement):**
1. `purchase_lead` — Days booked in advance
2. `length_of_stay` — Duration at destination
3. `booking_window_ratio` — Engineered ratio feature
4. `flight_hour` — Departure time
5. `sales_channel_num` — Mobile vs. Internet booking

---

### 6. Model 2 — XGBoost Classifier

XGBoost was trained on the same refined feature set (`X1`) for comparison:

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.15,
    scale_pos_weight=5,   # compensates for class imbalance
    random_state=42
)
xgb_model.fit(X1_train_scaled, y_train)
```

**Key Parameters:**
- `scale_pos_weight=5` — ratio of negative to positive class samples; equivalent to `class_weight='balanced'` in sklearn
- `learning_rate=0.15` — conservative step size for stable gradient descent

---

### 7. ROC-AUC Comparison

Both models were evaluated using the ROC-AUC metric — the most appropriate metric for imbalanced binary classification tasks, as it measures discriminative ability across all classification thresholds.

```python
rf_auc  = roc_auc_score(y_test, rf_probs)   # → 0.623
xgb_auc = roc_auc_score(y_test, xgb_probs)  # → 0.611
```

A **Precision-Recall Curve** was also plotted for the Random Forest model to assess performance specifically on the minority (booked) class.

---

## Final Results

| Model | ROC-AUC | Accuracy | Notes |
|---|---|---|---|
| **Random Forest** | **0.623** | ~73% (class 0) | ✅ Selected as final model |
| XGBoost | 0.611 | ~73% (class 0) | Slightly lower AUC |

**Winner: Random Forest** — achieved the highest ROC-AUC score of **0.623**, indicating a superior ability to discriminate between customers who will and will not complete a booking.

> The relatively modest AUC scores reflect the inherent difficulty of predicting human booking behavior. Incorporating richer behavioral data (e.g., browsing patterns, loyalty status, prior booking history) could meaningfully improve performance.

---

##  Testing the Model with New Data

Use the following code snippet to test the trained Random Forest model (`rf1_model`) on new, unseen customer data:

### Step 1 — Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost openpyxl
```

### Step 2 — Prepare New Data

New data must match the exact feature format used during training:

```python
import pandas as pd
import numpy as np

# Example: a single new customer record
new_data = pd.DataFrame({
    'length_of_stay':       [5],
    'flight_hour':          [14],
    'purchase_lead':        [30],
    'sales_channel_num':    [1],        # 0 = Mobile, 1 = Internet
    'trip_type_OneWay':     [0],        # 1 if One Way, else 0
    'trip_type_RoundTrip':  [1],        # 1 if Round Trip, else 0
    'booking_window_ratio': [30 / (5 + 1)]  # purchase_lead / (length_of_stay + 1)
})
```

### Step 3 — Scale and Predict

```python
# Scale using the SAME scaler fitted during training
new_data_scaled = scaler1.transform(new_data)

# Predict class label
prediction = rf1_model.predict(new_data_scaled)
print("Booking Prediction:", "✅ Will Complete Booking" if prediction[0] == 1 else "❌ Will NOT Complete Booking")

# Predict probability (confidence score)
probability = rf1_model.predict_proba(new_data_scaled)[:, 1]
print(f"Booking Probability: {probability[0]:.2%}")
```

### Step 4 — Apply Custom Threshold (Optional)

The default threshold is 0.5. Lower it to catch more potential bookers at the cost of more false positives:

```python
custom_threshold = 0.4
adjusted_prediction = (probability > custom_threshold).astype(int)
print("Adjusted Prediction:", "Will Complete Booking" if adjusted_prediction[0] == 1 else " Will NOT Complete Booking")
```

### Step 5 — Batch Predictions on Multiple Customers

```python
# Load a new CSV with multiple customers
new_customers = pd.read_csv("new_customers.csv")

# Engineer the booking_window_ratio feature
new_customers['booking_window_ratio'] = new_customers['purchase_lead'] / (new_customers['length_of_stay'] + 1)

# Encode sales_channel
new_customers['sales_channel_num'] = new_customers['sales_channel'].str.lower().str.strip().map({
    'mobile': 0, 'internet': 1
})

# One-hot encode trip_type
new_customers = pd.get_dummies(new_customers, columns=['trip_type'], drop_first=True, dtype=int)

# Select the correct feature columns
feature_cols = ['length_of_stay', 'flight_hour', 'purchase_lead', 
                'sales_channel_num', 'trip_type_OneWay', 'trip_type_RoundTrip',
                'booking_window_ratio']

X_new = new_customers[feature_cols]
X_new_scaled = scaler1.transform(X_new)

# Predict for all customers
predictions = rf1_model.predict(X_new_scaled)
probabilities = rf1_model.predict_proba(X_new_scaled)[:, 1]

new_customers['booking_prediction'] = predictions
new_customers['booking_probability'] = probabilities

print(new_customers[['booking_prediction', 'booking_probability']].head(10))
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.x` | Core programming language |
| `pandas` | Data manipulation and feature engineering |
| `numpy` | Numerical computations |
| `scikit-learn` | ML models, preprocessing, evaluation metrics |
| `xgboost` | Gradient boosting classification |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical visualizations |
| `openpyxl` | Reading Excel dataset |

---

## Project Structure

```
british-airways-booking-prediction/
│
├── BRITISH_AIRWAYS_JOB_SIMULATION.ipynb   # Main analysis notebook
├── customer_booking.xlsx                  # Dataset (not included — see note below)
└── README.md                              # Project documentation
```

> **Note:** The `customer_booking.xlsx` dataset is proprietary to the British Airways Forage Job Simulation. Download it directly from the [Forage platform](https://www.theforage.com/simulations/british-airways/data-science-yqoz).

---

## Key Takeaways

- **Class imbalance is a real challenge** in booking prediction — only ~15% of customers complete a booking. Proper handling via `class_weight='balanced'` and threshold tuning is essential.
- **Feature importance analysis** revealed that day-of-week dummies were largely noise; removing them and adding a domain-informed feature (`booking_window_ratio`) improved model interpretability.
- **purchase_lead** (days between booking and travel) was consistently the strongest predictor — customers who plan far ahead are more committed.
- **ROC-AUC over accuracy** — on imbalanced datasets, accuracy is misleading. A model predicting "never booked" would achieve ~85% accuracy but provide zero business value.
- **Random Forest outperformed XGBoost** marginally (AUC 0.623 vs 0.611) on this dataset, likely due to better ensemble averaging under class imbalance conditions.

---

## Acknowledgements

This project was completed as part of the **British Airways Data Science Virtual Job Simulation** hosted on [Forage](https://www.theforage.com/simulations/british-airways/data-science-yqoz). All business context and dataset belong to British Airways / Forage.

---

