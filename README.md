# Diabetes_Risk_Prediction
# Diabetes Risk Prediction

## 📌 Project Overview

This project aims to build a machine learning model to predict the risk of diabetes based on patient-related features such as age, BMI, sleep hours, and other health and lifestyle factors.

## 🎯 Project Goals

* Explore and understand the dataset.
* Perform data preprocessing and cleaning.
* Analyze relationships between features and diabetes risk.
* Prepare the data for machine learning models.
* Train and evaluate different classification models.
* Select the best-performing model.

## 📊 Dataset

The dataset contains patient information and features related to diabetes risk.

### Main Features

Some of the features include:

* Age
* BMI
* Hours of Sleep
* Other health and lifestyle-related features

The target variable represents whether the patient is at risk of diabetes.

## 🔎 Exploratory Data Analysis (EDA)

EDA was performed to understand:

* Dataset structure and feature types.
* Missing values.
* Duplicate records.
* Numerical and categorical features.
* Feature distributions.
* Relationships between features and the target variable.

## 🛠️ Data Preprocessing

The following preprocessing steps have been completed:

1. **Removed unnecessary columns**

   * Dropped columns that are not useful for prediction.

2. **Train-Test Split**

   * Split the data into training and testing sets.
   * The split was performed before fitting preprocessing steps to avoid data leakage.

3. **Missing Values**

   * Missing values were handled using appropriate imputation strategies.
   * The imputer was fitted on the training data and then applied to the test data.

4. **Categorical Encoding**

   * Categorical features were identified and encoded according to their type.
   * Nominal and ordinal categorical variables were handled appropriately.

5. **Feature Scaling**

   * Numerical features were scaled to prepare the data for machine learning models.

## 🚧 Current Progress

* [x] Load and inspect the dataset
* [x] Exploratory Data Analysis (EDA)
* [x] Identify numerical and categorical features
* [x] Remove unnecessary columns
* [x] Train-test split
* [x] Handle missing values
* [x] Encode categorical features
* [x] Scale numerical features
* [ ] Feature selection
* [ ] Train machine learning models
* [ ] Model evaluation
* [ ] Hyperparameter tuning
* [ ] Select the best model
* [ ] Final conclusions

## 🧰 Technologies & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

## 📌 Next Steps

The next stage of the project is **feature selection**, followed by training and comparing different machine learning classification models.
