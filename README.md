# Fraud Detection — Credit Card Transactions

## Overview

This project builds a binary classification model to predict whether a credit card transaction is fraudulent using a Random Forest Classifier.

The dataset was sourced from Kaggle's Credit Card Fraud Detection dataset. The core objective was to find the right balance between false negatives (missed fraud) and false positives (false alarms).

---

## Problem

Given credit card transaction data, predict whether a transaction is fraudulent based on:

- Transaction amount
- Merchant category
- Cardholder demographics (age, gender)
- Location context (city population)

---

## Approach

### 1. Data Preprocessing and Feature Engineering

The original dataset contained 142,681 records, with only 0.39% of transactions labeled as fraudulent. To address this severe class imbalance, undersampling was applied before splitting the data — retaining all fraudulent transactions and sampling 2,500 non-fraudulent records to create a more balanced dataset.

An `age` feature was engineered by calculating the difference between `trans_date_trans_time` and `dob`, representing the cardholder's age at the time of each transaction.

---

### 2. Pipeline

A `ColumnTransformer` pipeline was built with separate transformers for numeric and categorical features:

- **Numeric features** (`amt`, `city_pop`, `age`): Missing values imputed using `median`. Scaled with `RobustScaler` due to severe outliers in transaction amounts.
- **Categorical features** (`category`, `gender`): Missing values imputed using `most_frequent`. Encoded with `OneHotEncoder(handle_unknown='ignore')` to handle unseen categories during prediction without errors.

---

### 3. Model

- Model: `RandomForestClassifier(class_weight='balanced')`
- `class_weight='balanced'` applied to further address class imbalance
- Goal: Classify transactions as fraudulent or not

---

### 4. Hyperparameter Tuning

`GridSearchCV` was used to find the optimal number of trees, testing `n_estimators` values of 100, 200, 300, 400, and 500. `recall` was used as the scoring metric because missing a fraudulent transaction (false negative) is more costly than a false alarm.

**Best parameter:** `n_estimators = 500`

---

### 5. Threshold Tuning

Thresholds from 0.50 down to 0.10 were tested to find the optimal balance between false negatives and false positives.

| Threshold | TN | FP | FN | TP |
|-----------|----|----|----|----|
| 0.50 | 603 | 22 | 26 | 511 |
| 0.45 | 600 | 25 | 24 | 513 |
| 0.40 | 597 | 28 | 23 | 514 |
| 0.35 | 588 | 37 | 19 | 518 |
| 0.30 | 583 | 42 | 12 | 525 |
| 0.25 | 577 | 48 | 11 | 526 |
| **0.20** | **567** | **58** | **6** | **531** |
| 0.15 | 551 | 74 | 3 | 534 |
| 0.10 | 519 | 106 | 0 | 537 |

**Final threshold: 0.20** — false negatives dropped from 26 to 6 at the cost of increasing false positives from 22 to 58. This tradeoff is acceptable — undetected fraud directly harms customers and carries a higher long-term business cost than routing additional transactions to manual review.

---

## Evaluation

- **Final Threshold:** 0.20
- **FN:** 6
- **FP:** 58

### Interpretation

- The model catches the majority of fraudulent transactions at the chosen threshold
- Remaining false positives are routed to manual review — an acceptable operational cost

---

## Key Insights

Transaction amount (`amt`) is by far the strongest predictor of fraud at **66.8%** feature importance.

| Feature | Importance |
|---------|------------|
| amt | 66.8% |
| age | 7.06% |
| city_pop | 7.01% |

High-value transactions are the primary fraud signal. The engineered `age` feature also contributed meaningfully alongside city population.

---

## Limitations

### Class Imbalance
- Original dataset is severely imbalanced (0.39% fraud)
- Undersampling was used but reduces available training data

### Threshold Sensitivity
- Results vary across different random splits
- Best `n_estimators` and optimal threshold may shift with different data samples

---

## Future Improvements

- Apply SMOTE for synthetic oversampling instead of undersampling
- Test XGBoost for comparison
- Add more features (e.g., transaction frequency, time of day)
- Evaluate on the full dataset without undersampling

---

## Tech Stack

- Python (pandas, numpy, scikit-learn)
- RandomForestClassifier
- GridSearchCV, Pipeline, ColumnTransformer
