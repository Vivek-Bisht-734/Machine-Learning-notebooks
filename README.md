# Machine-Learning-notebooks
Collection of data science notebooks &amp; many small projects for different models

# Project 1: Sonar-Rock-Mine-Prediction

**Short description:** A compact project to train and evaluate a classifier (Logistic Regression) on the Sonar dataset to predict whether a sonar return is from a **rock** or a **mine** (metal cylinder). Intended as a reproducible demo and reference notebook for submarine sonar classification.
This repository contains a reproducible notebook that builds a logistic regression classifier on the Sonar dataset to predict whether a sonar signal corresponds to a rock or a metal mine. The notebook includes data loading, exploratory data analysis, preprocessing, model training, evaluation (accuracy_score).

### Dataset
* **Source:** Sonar dataset (Rocks vs Mines)
* **Columns:** 60 numeric attributes (sonar measurements) + 1 label column (`R` or `M` / `Rock` or `Mine`).

### Environment & Requirements
pandas
numpy
scikit-learn

-------------------------------------------------------------------------------------------------------------

# Project 2: Online-Payments-Fraud-Detection

**Short description:** A compact, reproducible project that builds and evaluates a machine-learning classifier to detect fraudulent online transactions. 
The notebook explores the dataset, performs preprocessing (encoding, missing value handling, feature selection), trains a baseline model (Decision Tree), evaluates using standard classification metrics, and includes a small prediction example.

### Dataset
**Source:** Transaction-level data containing fields such as step, type (TRANSFER, PAYMENT, CASH_OUT, ...), amount, account balances (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest), and isFraud (0/1).
**Columns:** step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

### Environment & Requirements
pandas
numpy
scikit-learn
matplotlib.pyplot

-------------------------------------------------------------------------------------------------------------



## Contact
mailto:vivekbisht0270@gmail.com
