# Customer Churn Prediction System

## Overview

This project builds an end-to-end machine learning system to predict whether a bank customer is likely to churn (leave the bank). The solution leverages classification models, feature engineering, and model evaluation techniques, and is deployed using an interactive web application.

## Problem Statement

Customer churn is a major challenge in the banking sector. Retaining existing customers is significantly more cost-effective than acquiring new ones.

 **Goal:** Predict customers at risk of churning and enable proactive retention strategies.

## Dataset

* Bank Customer Churn Dataset (Kaggle)
* Contains customer demographics, account details, and engagement metrics

### Key Features:

* Credit Score, Age, Balance
* Number of Products
* Active Membership Status
* Geography & Gender

### Target:

* `Churn` (1 = Churn, 0 = Stay)

## Project Workflow

1. **Data Cleaning**

   * Removed irrelevant columns (CustomerId, Surname, etc.)
   * Handled missing values

2. **Exploratory Data Analysis (EDA)**

   * Churn distribution analysis
   * Customer behavior patterns

3. **Feature Engineering**

   * Balance per product
   * Activity score
   * Age group segmentation

4. **Handling Imbalance**

   * Applied SMOTE to balance classes

5. **Model Training**

   * Logistic Regression (baseline)
   * Random Forest (final model)

6. **Model Evaluation**

   * Accuracy, Confusion Matrix
   * ROC-AUC Score
   * Classification Report

7. **Deployment**

   * Built an interactive web app using Streamlit


## Results

| Metric         | Value  |
| -------------- | ------ |
| Accuracy       | 84.75% |
| ROC-AUC        | 0.85   |
| Recall (Churn) | 58%    |

The model performs well overall but highlights challenges in detecting churn cases.



## Insights

* Inactive customers are more likely to churn
* Customers with fewer products show higher churn risk
* Older customers tend to churn more
* Customer engagement is a stronger predictor than balance


## Streamlit App

An interactive app allows users to:

* Input customer details
* Predict churn probability
* Get instant results

### Run Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

* Improve churn recall using threshold tuning
* Add XGBoost / Gradient Boosting
* Deploy app on cloud (Streamlit Cloud / AWS)
* Add explainability (SHAP values)
