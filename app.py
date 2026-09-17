import streamlit as st
import pandas as pd
import joblib

# Load models
logistic = joblib.load("models/logistic.pkl")
lasso = joblib.load("models/lasso.pkl")
ridge = joblib.load("models/ridge.pkl")
decision_tree = joblib.load("models/decision_tree.pkl")
random_forest = joblib.load("models/random_forest.pkl")


# Load preprocessing objects
num_imputer = joblib.load("preprocessing/num_imputer.pkl")
cat_imputer = joblib.load("preprocessing/cat_imputer.pkl")
encoder = joblib.load("preprocessing/encoder.pkl")
scaler = joblib.load("preprocessing/scaler.pkl")
mappings = joblib.load("preprocessing/mappings.pkl")
feature_columns = joblib.load("preprocessing/feature_columns.pkl")


st.title("🩺 Diabetes Risk Prediction")
st.write("Enter patient information to predict diabetes risk.")


age = st.number_input("Age", min_value=1.0, max_value=120.0, value=30.0)

bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)

hours_sleep = st.number_input(
    "Hours Sleep Per Night",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

stress_level = st.number_input(
    "Stress Level",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

fasting_blood_sugar = st.number_input(
    "Fasting Blood Sugar",
    min_value=0.0,
    max_value=500.0,
    value=100.0
)

hba1c_level = st.number_input(
    "HbA1c Level",
    min_value=0.0,
    max_value=20.0,
    value=5.5
)

systolic = st.number_input(
    "Blood Pressure Systolic",
    min_value=50.0,
    max_value=250.0,
    value=120.0
)

diastolic = st.number_input(
    "Blood Pressure Diastolic",
    min_value=30.0,
    max_value=150.0,
    value=80.0
)

waist = st.number_input(
    "Waist Circumference (cm)",
    min_value=30.0,
    max_value=200.0,
    value=90.0
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

city = st.selectbox(
    "City",
    [
        'Mumbai',
        'Thane',
        'Bengaluru',
        'Hyderabad',
        'Delhi',
        'Kolkata',
        'Ahmedabad',
        'Pune',
        'Chennai',
        'Kanpur',
        'Bhopal',
        'Visakhapatnam',
        'Patna',
        'Jaipur',
        'Surat',
        'Nagpur',
        'Indore',
        'Lucknow'
    ]
)

family_history = st.selectbox(
    "Family History of Diabetes",
    ["No", "Yes"]
)

activity = st.selectbox(
    "Physical Activity Level",
    ["Sedentary", "Moderate", "Active"]
)

diet = st.selectbox(
    "Diet Type",
    ["Non-Vegetarian", "Vegetarian", "Pescatarian", "Vegan"]
)

smoking = st.selectbox(
    "Smoking Status",
    ["Never", "Current", "Former"]
)

alcohol = st.selectbox(
    "Alcohol Consumption",
    ["Never", "Occasional", "Regular"]
)

income = st.selectbox(
    "Income Bracket",
    ["Low", "Middle", "High"]
)


model_name = st.selectbox(
        "Choose Model",
        [
            "Logistic Regression",
            "Lasso",
            "Ridge",
            "Decision Tree",
            "Random Forest"
        ]
    )


if st.button("Predict Diabetes Risk"):
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "hours_sleep_per_night": [hours_sleep],
        "stress_level": [stress_level],
        "fasting_blood_sugar": [fasting_blood_sugar],
        "hba1c_level": [hba1c_level],
        "blood_pressure_systolic": [systolic],
        "blood_pressure_diastolic": [diastolic],
        "waist_circumference_cm": [waist],

        "gender": [gender],
        "city": [city],
        "family_history_diabetes": [family_history],
        "physical_activity_level": [activity],
        "diet_type": [diet],
        "smoking_status": [smoking],
        "alcohol_consumption": [alcohol],
        "income_bracket": [income]
    })

    numeric_features = [
        "age",
        "bmi",
        "hours_sleep_per_night",
        "stress_level",
        "fasting_blood_sugar",
        "hba1c_level",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "waist_circumference_cm"
    ]

    input_data[numeric_features] = num_imputer.transform(
        input_data[numeric_features]
    )

    categorical_features = [
        "gender",
        "city",
        "family_history_diabetes",
        "physical_activity_level",
        "diet_type",
        "smoking_status",
        "alcohol_consumption",
        "income_bracket"
    ]

    input_data[categorical_features] = cat_imputer.transform(
        input_data[categorical_features]
    )

    input_data["physical_activity_level"] = (
        input_data["physical_activity_level"]
        .map(mappings["activity"])
    )

    input_data["alcohol_consumption"] = (
        input_data["alcohol_consumption"]
        .map(mappings["alcohol"])
    )

    input_data["income_bracket"] = (
        input_data["income_bracket"]
        .map(mappings["income"])
    )

    input_data["family_history_diabetes"] = (
        input_data["family_history_diabetes"]
        .map(mappings["binary"])
    )

    nominal_cols = [
        "gender",
        "city",
        "diet_type",
        "smoking_status"
    ]

    encoded = encoder.transform(
        input_data[nominal_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(nominal_cols)
    )

    input_data = input_data.drop(
        columns=nominal_cols
    )

    input_data = pd.concat(
        [input_data, encoded_df],
        axis=1
    )

    input_data = input_data[feature_columns]


    input_scaled = input_data.copy()

    input_scaled[numeric_features] = scaler.transform(
        input_data[numeric_features]
    )

    if model_name == "Logistic Regression":

        prediction = logistic.predict(input_scaled)

    elif model_name == "Lasso":

        prediction = lasso.predict(input_scaled)

    elif model_name == "Ridge":

        prediction = ridge.predict(input_scaled)

    elif model_name == "Decision Tree":

        prediction = decision_tree.predict(input_data)

    else:

        prediction = random_forest.predict(input_data)

    st.success(
    f"Predicted Diabetes Risk: {prediction[0]}"
    )

    if model_name == "Logistic Regression":

        prediction = logistic.predict(input_scaled)
        probabilities = logistic.predict_proba(input_scaled)
        classes = logistic.classes_

    elif model_name == "Lasso":

        prediction = lasso.predict(input_scaled)
        probabilities = lasso.predict_proba(input_scaled)
        classes = lasso.classes_

    elif model_name == "Ridge":

        prediction = ridge.predict(input_scaled)
        probabilities = ridge.predict_proba(input_scaled)
        classes = ridge.classes_

    elif model_name == "Decision Tree":

        prediction = decision_tree.predict(input_data)
        probabilities = decision_tree.predict_proba(input_data)
        classes = decision_tree.classes_

    else:

        prediction = random_forest.predict(input_data)
        probabilities = random_forest.predict_proba(input_data)
        classes = random_forest.classes_

    # st.success(
    #     f"Predicted Diabetes Risk: {prediction[0]}"
    # )

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame(
        [probabilities[0]],
        columns=classes
    )

    st.dataframe(probability_df)

