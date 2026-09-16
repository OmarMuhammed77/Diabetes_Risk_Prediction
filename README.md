# Diabetes Risk Prediction 🩺

A Machine Learning classification project that predicts a patient's **diabetes risk level** as **Low, Moderate, or High** based on demographic, lifestyle, and health-related features.

## 📌 Project Overview

The goal of this project is to build and compare several machine learning classification models for predicting diabetes risk.

The project covers the main stages of a typical Machine Learning workflow:

* Data Understanding
* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Missing Value Handling
* Feature Encoding
* Feature Scaling
* Feature Selection
* Model Training
* Model Evaluation
* Model Comparison

## 📊 Dataset

The dataset contains patient-related information such as:

* Age
* BMI
* Fasting Blood Sugar
* HbA1c Level
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Waist Circumference
* Sleep Hours
* Stress Level
* Physical Activity Level
* Alcohol Consumption
* Income Bracket
* Family History of Diabetes
* Gender
* City
* Diet Type
* Smoking Status

### Target Variable

`diabetes_risk`

The target contains three classes:

* **Low**
* **Moderate**
* **High**

The classes are imbalanced, with the Low-risk class having the largest number of samples.

## 🔍 Exploratory Data Analysis

EDA was performed to understand the structure and distribution of the dataset.

The analysis included:

* Dataset shape and statistical summary
* Data types and unique values
* Missing value analysis
* Target distribution
* Distribution of numerical features
* Outlier detection using boxplots

The target distribution was visualized to understand the number and percentage of patients in each diabetes-risk category.

## 🧹 Data Preprocessing

### 1. Removing Unnecessary Columns

The following columns were removed:

* `patient_id`
* `diabetes_risk` from the feature set

The target variable was stored separately as `y`.

### 2. Train-Test Split

The dataset was divided into:

* **80% Training data**
* **20% Testing data**

Stratified splitting was used to preserve the class distribution:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 3. Missing Values

Missing numerical values were handled using **median imputation**.

Missing categorical values were handled using **most-frequent imputation**.

The imputers were fitted only on the training data and then applied to the test data to avoid data leakage.

### 4. Feature Encoding

Different encoding techniques were used depending on the type of categorical variable.

#### Ordinal Encoding

The following ordered features were converted into numerical values:

* `physical_activity_level`
* `alcohol_consumption`
* `income_bracket`

#### Binary Encoding

`family_history_diabetes` was converted into:

* `No → 0`
* `Yes → 1`

#### One-Hot Encoding

Nominal categorical features were encoded using `OneHotEncoder`:

* Gender
* City
* Diet Type
* Smoking Status

Unknown categories were handled using:

```python
handle_unknown='ignore'
```

### 5. Feature Scaling

`StandardScaler` was used to standardize the numerical features for models that benefit from scaling, particularly Logistic Regression.

The scaler was fitted on the training data and then used to transform the test data.

## 🎯 Feature Selection

**Mutual Information (MI)** was used to measure the relationship between each feature and the target variable.

The MI scores were calculated using:

```python
mutual_info_classif(
    x_train,
    y_train,
    random_state=32
)
```

The resulting features were ranked according to their Mutual Information scores and visualized using a bar chart.

> Mutual Information was used to analyze feature relevance. The current implementation did not remove features based on a selected MI threshold.

## 🤖 Machine Learning Models

Five classification approaches were trained and evaluated:

1. Logistic Regression
2. Lasso Logistic Regression
3. Ridge Logistic Regression
4. Decision Tree
5. Random Forest

### Logistic Regression

Standard Logistic Regression was trained using scaled features.

### Lasso Logistic Regression

L1 regularization was used with:

```python
penalty='l1'
C=0.1
```

### Ridge Logistic Regression

L2 regularization was used with:

```python
penalty='l2'
C=0.1
```

### Decision Tree

The Decision Tree was configured with:

```python
max_depth=5
min_samples_split=10
min_samples_leaf=5
```

### Random Forest

The Random Forest model was configured with:

```python
n_estimators=100
```

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

Macro averaging was used because the target contains three classes with an imbalanced distribution.

## 🏆 Model Comparison

| Model                     |   Accuracy |       Precision |       Recall |         F1 |
| ------------------------- | ---------: | --------------: | -----------: | ---------: |
| Logistic Regression       | **79.67%** |      **76.11%** |   **73.37%** | **74.60%** |
| Lasso Logistic Regression |     77.23% |          72.85% |       69.01% |     69.34% |
| Ridge Logistic Regression |     79.53% |          76.07% |       73.07% |     74.42% |
| Decision Tree             |     76.50% |          73.03% |       68.76% |     70.51% |
| Random Forest             |     77.87% |          74.02% |       70.99% |     72.34% |

The results show the performance of each model on the held-out test set.

## 🛠️ Technologies & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

## 📁 Project Structure

```text
Diabetes-Risk-Prediction/
│
├── diabetes_risk.csv
├── Diabetes_Risk_Prediction.ipynb
└── README.md
```

## 🚀 Machine Learning Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
EDA
   ↓
Train / Test Split
   ↓
Missing Value Handling
   ↓
Feature Encoding
   ↓
Feature Scaling
   ↓
Feature Selection Analysis
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
```

## 📌 Notes

* The project uses a train/test split with stratification.
* Missing-value imputers and the scaler are fitted on the training data only.
* Macro Precision, Macro Recall, and Macro F1 are used to account for the multi-class target and class imbalance.
* Cross-validation was **not used** in the current version.
* Hyperparameter tuning was **not used** in the current version.

## 🔮 Future Improvements

Possible improvements for a future version include:

* Cross-validation
* Hyperparameter tuning
* More advanced feature selection
* Testing additional classification algorithms
* Handling class imbalance using appropriate techniques
* More detailed confusion-matrix analysis
* Model interpretation and feature importance analysis
* Deployment of the final model as an application or API

