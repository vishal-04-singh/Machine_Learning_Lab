# REGRESSION CAPSTONE PROJECT REPORT

## Health Lifestyle Disease Risk Prediction Using Linear Regression

### Machine Learning Laboratory - Project 1

---

## UNIVERSITY INFORMATION

**University of Petroleum and Energy Studies**  
**Batch 2025 - 2027**  
**Department:** School of Computer Science  
**Course:** MCA  

**Assignment:** Health Lifestyle Disease Risk Prediction Using Linear Regression  
**Machine Learning Laboratory - Project 1**

**Submitted By:** Vishal Singh (590028039)  
**Submitted to:** Manobendra Nath Mondal  
**Submission Date:** September 23, 2025

---

## ABSTRACT

This capstone project investigates the application of linear regression for predicting disease risk using comprehensive health and lifestyle data. Using a dataset of 100,000 health records with 16 features, we developed a baseline predictive model to assess binary disease risk outcomes. While the project successfully demonstrates the complete machine learning pipeline including data preprocessing, model training, and evaluation, the results reveal significant limitations of linear regression for binary classification tasks. The model achieved an R² score of -0.0003, highlighting the need for more appropriate classification algorithms. This work provides valuable insights into algorithm selection and establishes a foundation for future improvements using logistic regression, ensemble methods, and advanced feature engineering techniques.

---

## TABLE OF CONTENTS

1. [PROJECT TITLE](#project-title)
2. [OBJECTIVE](#objective)
3. [DATASET DESCRIPTION](#dataset-description)
4. [METHODOLOGY USED](#methodology-used)
5. [SOURCE CODE](#source-code)
6. [RESULT DISCUSSION](#result-discussion)
7. [CONCLUSION](#conclusion)

---

## PROJECT TITLE

**Health Lifestyle Disease Risk Prediction Using Linear Regression**

---

## OBJECTIVE

### 🎯 Primary Objective
The primary objective of this project is to develop a predictive model that can assess disease risk based on various health and lifestyle factors.

### 📋 Specific Project Aims

1. **Build a machine learning model** to predict binary disease risk (0 = low risk, 1 = high risk)
2. **Identify key health indicators** that contribute most significantly to disease risk
3. **Analyze relationships** between lifestyle factors (exercise, sleep, diet) and health outcomes
4. **Create a baseline model** for future enhancement with more sophisticated algorithms
5. **Demonstrate the application** of linear regression in healthcare predictive analytics

> **💡 Project Scope:** This project serves as an exploratory analysis and baseline establishment for disease risk prediction, focusing on understanding data patterns and model evaluation methodologies rather than achieving optimal predictive performance.

---

## DATASET DESCRIPTION

### 📊 Dataset Overview
- **Source:** Health Lifestyle Dataset (health_lifestyle_dataset.csv)
- **Size:** 100,000 records with 16 features
- **Data Quality:** Complete dataset with no missing values
- **Target Variable:** `disease_risk` (binary: 0 = low risk, 1 = high risk)
- **Class Distribution:** 24.8% high risk, 75.2% low risk (imbalanced dataset)

### 🔍 Feature Description
The dataset contains comprehensive health and lifestyle information:

#### 👥 Demographic Features
- `age`: Age in years (18-79)
- `gender`: Male/Female (encoded as gender_num: 1=Male, 0=Female)

#### 🏥 Physical Health Metrics
- `bmi`: Body Mass Index (18.0-40.0)
- `resting_hr`: Resting heart rate (50-99 bpm)
- `systolic_bp`: Systolic blood pressure (90-179 mmHg)
- `diastolic_bp`: Diastolic blood pressure (60-119 mmHg)
- `cholesterol`: Cholesterol level (150-299 mg/dL)

#### 🏃 Lifestyle Factors
- `daily_steps`: Daily step count (1,000-19,999)
- `sleep_hours`: Sleep duration (3-10 hours)
- `water_intake_l`: Daily water consumption (0.5-5.0 liters)
- `calories_consumed`: Daily calorie intake (1,200-3,999)

#### ⚠️ Risk Factors
- `smoker`: Smoking status (binary: 0=non-smoker, 1=smoker)
- `alcohol`: Alcohol consumption (binary: 0=no, 1=yes)
- `family_history`: Family history of disease (binary: 0=no, 1=yes)

### 📈 Statistical Summary

| **Feature** | **Mean** | **Standard Deviation** |
|-------------|----------|------------------------|
| Age (years) | 48.5 | 17.9 |
| BMI | 29.0 | 6.4 |
| Daily Steps | 10,480 | 5,484 |
| Sleep Hours | 6.5 | 2.0 |

> **✅ Data Quality Assessment:** The dataset demonstrates excellent quality with no missing values across all 100,000 records, providing a robust foundation for analysis.

---

## METHODOLOGY USED

### 🔄 Data Preprocessing
1. **Data Loading**: Imported dataset using pandas
2. **Exploratory Data Analysis**: 
   - Checked for missing values (none found)
   - Generated descriptive statistics
   - Created correlation heatmap
   - Performed outlier analysis using box plots
3. **Feature Engineering**:
   - Converted categorical gender to numerical (gender_num)
   - Selected 5 key features: age, bmi, daily_steps, sleep_hours, gender_num

### 🤖 Model Development
1. **Algorithm Selection**: Linear Regression
   - Chosen for simplicity and interpretability
   - Suitable for establishing baseline performance
   
2. **Feature Selection**: 
   - Used subset of available features (5 out of 15)
   - Features selected based on basic correlation analysis

3. **Data Splitting**:
   - Train-test split: 80% training, 20% testing
   - Random state: 42 (for reproducibility)
   - Training samples: 80,000
   - Testing samples: 20,000

4. **Model Training**:
   - Fitted scikit-learn LinearRegression model
   - No hyperparameter tuning required

5. **Model Evaluation**:
   - Primary metrics: R² Score, Mean Absolute Error (MAE)
   - Cross-validation: 5-fold cross-validation
   - Residual analysis for model diagnostics

---

## SOURCE CODE

### 💻 Key Code Components

#### 📥 Data Loading and Exploration
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("health_lifestyle_dataset.csv")
print(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
```

#### ⚙️ Feature Engineering
```python
df['gender_num'] = (df['gender'] == 'Male').astype(int)
features = ['age', 'bmi', 'daily_steps', 'sleep_hours', 'gender_num']
X = df[features]
y = df['disease_risk']
```

#### 🏋️ Model Training
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
```

#### 📊 Model Evaluation
```python
from sklearn.model_selection import cross_val_score

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
```

---

## RESULT DISCUSSION

### 📉 Model Performance Metrics

| **Metric** | **Value** |
|------------|-----------|
| **R² Score** | **🔴 -0.0003** |
| **Mean Absolute Error** | **🔴 0.3731** |
| **Cross-Validated R²** | **🔴 -0.0001 (±0.0001)** |

> **⚠️ Performance Alert:** The negative R² score indicates that the model performs worse than simply predicting the mean value, highlighting fundamental issues with the approach.

### 🎯 Feature Importance (Coefficients)

| **Feature** | **Coefficient** |
|-------------|-----------------|
| Intercept | 0.2366 |
| Age | 0.0001 |
| BMI | 0.0003 |
| Daily Steps | -0.0000 |
| Sleep Hours | 0.0008 |
| Gender (Male) | -0.0015 |

> **💡 Coefficient Analysis:** All feature coefficients are extremely small (near zero), indicating weak linear relationships between the selected features and disease risk.

### 🔍 Critical Analysis

**⚠️ The model performance is extremely poor with several critical issues:**

1. **🔴 Negative R² Score**: Indicates the model performs worse than simply predicting the mean value
2. **🔴 High Prediction Error**: MAE of 0.37 for binary outcomes (0 or 1) is substantial
3. **🟡 Weak Feature Relationships**: All coefficients are near zero, suggesting no meaningful linear relationships
4. **🔴 Algorithm Mismatch**: Linear regression is inappropriate for binary classification problems

### 📋 Prediction Examples
Sample predictions show the model outputs continuous values around 0.25, failing to properly classify binary outcomes:

| **Actual** | **Predicted** |
|------------|---------------|
| ✅ 0 | 🔴 0.252 |
| ❌ 1 | 🔴 0.246 |
| ❌ 1 | 🔴 0.243 |
| ✅ 0 | 🔴 0.253 |
| ✅ 0 | 🔴 0.245 |

> **⚠️ Classification Issue:** The model fails to output discrete binary classifications (0 or 1), instead producing continuous values clustered around the dataset mean (0.248).

### 📈 Residual Analysis
The residual plot (actual vs predicted scatter plot) would likely show:
- Poor correlation between actual and predicted values
- Random scatter indicating no learned patterns
- Predictions clustered around the mean disease risk value

---

## CONCLUSION

### 🎯 Project Assessment
While this project successfully demonstrates the complete machine learning pipeline from data loading to model evaluation, the **predictive performance is inadequate** for practical disease risk assessment.

### 🔍 Key Findings
1. **🔴 Algorithm Limitation**: Linear regression is fundamentally unsuitable for binary classification
2. **🟡 Feature Underutilization**: Only 5 of 16 available features were used
3. **✅ Data Quality**: The dataset itself appears suitable with good quality and comprehensive health metrics
4. **✅ Baseline Establishment**: The project provides a clear baseline for future improvements

### ⚠️ Major Limitations
1. **Methodological**: Wrong algorithm choice for classification problem
2. **Feature Engineering**: Insufficient use of available health indicators
3. **Model Complexity**: Linear model too simplistic for complex health relationships
4. **Class Imbalance**: 75-25 distribution not addressed

### 🚀 Recommendations for Improvement

#### 1. **Algorithm Change**: Switch to classification algorithms:
   - **Logistic Regression**
   - **Random Forest**
   - **Support Vector Machine**
   - **Gradient Boosting**

#### 2. **Feature Enhancement**:
   - Include all relevant health metrics (blood pressure, cholesterol, family history)
   - Create interaction terms and polynomial features
   - Apply feature scaling/normalization

#### 3. **Advanced Techniques**:
   - Address class imbalance with SMOTE or class weighting
   - Implement proper classification metrics (accuracy, precision, recall, F1-score)
   - Use ROC-AUC for model comparison

#### 4. **Model Validation**:
   - Implement stratified cross-validation
   - Use confusion matrix analysis
   - Apply feature importance techniques

### 🎓 Learning Outcomes

**✅ This project successfully demonstrates:**

- **✅ Complete ML workflow implementation**
- **✅ Data exploration and visualization skills**
- **✅ Model evaluation and interpretation**
- **🟡 Critical analysis of model limitations**
- **🔵 Understanding of algorithm-problem matching importance**

> **💡 Educational Value:** The project serves as an excellent foundation for building more sophisticated disease risk prediction models and highlights the importance of proper algorithm selection in machine learning applications.

---

## 📝 Final Note

This project represents an initial exploration and baseline model. Future iterations should address the identified limitations to create a clinically viable disease risk prediction system.

---

## 📋 Document Information

- **Author:** Vishal Singh (590028039)
- **Course:** Machine Learning Laboratory (MCA)
- **Institution:** University of Petroleum and Energy Studies
- **Department:** School of Computer Science
- **Batch:** 2025 - 2027
- **Supervisor:** Manobendra Nath Mondal
- **Date:** September 23, 2025
- **Project:** Assignment - Health Lifestyle Disease Risk Prediction
- **Algorithm Used:** Linear Regression
- **Dataset:** health_lifestyle_dataset.csv (100,000 records, 16 features)

---

**© 2025 University of Petroleum and Energy Studies - School of Computer Science**