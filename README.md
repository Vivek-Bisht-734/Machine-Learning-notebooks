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

# Project 3: Waiter-Tips-Prediction

**Short description:** A predictive modeling project that estimates the tip amount a waiter is likely to receive from customers based on bill characteristics and other influencing factors. The notebook includes exploratory data analysis, feature visualization, preprocessing, model training (Linear Regression), and evaluation using standard regression metrics mean_squared_error, mean_absolute_error, r2_score(MSE, MAE, R²).

### Dataset
**Source:** Tips dataset from kaggle.
**Columns:**
total_bill – Total bill amount
tip – Tip amount (target variable)
sex – Gender of customer
smoker – Whether the customer is a smoker (Yes/No)
day – Day of the week (Thur, Fri, Sat, Sun)
time – Lunch/Dinner
size – Number of people at the table

### Environment & Requirements
pandas
numpy
scikit-learn

-------------------------------------------------------------------------------------------------------------

# Project 4: Diabetes-Prediction

**Short description:** A compact machine learning project that builds a classification model to predict whether a person has diabetes or not based on medical diagnostic data. The notebook includes dataset exploration, preprocessing (handling missing/invalid values, scaling), training a baseline model (Support Vector Machine), evaluation using classification metrics (accuracy_score), and prediction examples.

### Dataset
**Source:** PIMA Indians Diabetes dataset (kaggle and dropbox).
**Columns:**
Pregnancies – Number of pregnancies
Glucose – Plasma glucose concentration
BloodPressure – Diastolic blood pressure
SkinThickness – Triceps skinfold thickness
Insulin – 2-Hour serum insulin
BMI – Body mass index
DiabetesPedigreeFunction – Diabetes pedigree function (genetic influence)
Age – Age of the person
Outcome – 1: Diabetic, 0: Non-diabetic (target variable)

### Environment & Requirements
pandas
numpy
scikit-learn

-------------------------------------------------------------------------------------------------------------


## Contact
mailto:vivekbisht0270@gmail.com
