# 🛡️ ChurnShield AI  
## Customer Churn Prediction with Interpretable Machine Learning

Customer churn prediction is a core business problem in subscription-based industries. The real value of a churn model lies not only in prediction accuracy, but in its ability to **identify risk early and explain why customers leave**.

**ChurnShield AI** is an end-to-end machine learning project that predicts customer churn using telecom customer data and presents results through a clean, interactive web application built with Streamlit.

---

## 🚀 Live Application

🔗 **Streamlit App:**  
https://customer-churn-risk-predictor.streamlit.app/

---

## 📌 Project Summary

This project is based on the **IBM Telco Customer Churn dataset**, a widely used real-world dataset that captures customer demographics, service usage, and billing information.

- **Problem Type:** Binary classification (Churn vs No Churn)
- **Domain:** Telecommunications
- **Objective:** Identify customers at risk of churning to support retention strategies
- **Approach:** Emphasis on interpretability, robustness, and deployment readiness

---

## 📂 Dataset Overview

- Each row represents a single customer
- Target column: `Churn`
- Key feature groups:
  - Customer tenure and contract details
  - Service subscriptions (internet, streaming, phone)
  - Billing and payment behavior
  - Demographic indicators

---

## 🔍 Exploratory Data Analysis

- Analyzed churn distribution and class imbalance
- Identified high-risk segments based on contract type and services
- Visualized tenure, charges, and churn relationships
- Used automated profiling for initial understanding and manual EDA for business insights

---

## 🧠 Feature Engineering & Preprocessing

- Created **AvgCharges** to better represent long-term customer value
- Encoded categorical variables using one-hot encoding
- Scaled numerical features using `StandardScaler`
- Ensured no target leakage during feature creation

---

## 🧪 Model Development & Evaluation

Multiple models were trained and evaluated using **ROC-AUC**, a suitable metric for imbalanced churn data.

| Model | ROC-AUC |
|-----|--------|
| Logistic Regression (L1) | **0.845** |
| Decision Tree | 0.832 |
| Random Forest | 0.843 |
| XGBoost | 0.840 |

---

## 🏆 Final Model Selection

**Logistic Regression with L1 regularization** was selected as the final model because:

- It achieved the best ROC-AUC among tested models
- It provided stable performance with lower complexity
- L1 regularization enabled feature selection
- Model coefficients offered clear business interpretability

This indicates that churn patterns in this dataset are largely linear and well captured through careful feature engineering.

---

## 📊 Key Insights

### Lower Churn Risk
Customers tend to stay longer when they:
- Have long tenure
- Are on one- or two-year contracts
- Show consistent spending behavior

### Higher Churn Risk
Customers are more likely to churn when they:
- Are on month-to-month contracts
- Use fiber optic internet services
- Pay via electronic check

These insights can directly inform retention and pricing strategies.

---

## 🚀 Deployment

- Built an interactive Streamlit application
- Loads trained model and scaler using `joblib`
- Accepts real-time user inputs and generates churn risk scores
- Designed with a business-friendly layout for non-technical users

---

## 🛠️ Technology Stack

**Data & Analysis**
- Pandas, NumPy

**Visualization**
- Matplotlib, Seaborn

**Machine Learning**
- Scikit-learn (Logistic Regression, RFE, StandardScaler)
- XGBoost

**Deployment**
- Streamlit
- Joblib

---

## ⚙️ Running the Project Locally

```bash
git clone https://github.com/Sadat-Shakeeb/customer-churn-prediction-ml.git
cd customer-churn-prediction-ml
pip install -r requirements.txt
streamlit run app.py

## 📌 Key Takeaway
This project demonstrates a complete machine learning workflow — from data understanding and feature engineering to model selection and deployment — with a strong focus on interpretability and real-world usability.
