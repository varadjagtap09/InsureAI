import streamlit as st
import pandas as pd
import joblib
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="InsureAI",
)

st.title("InsureAI - Insurance Cost Prediction")

st.write(
    "Enter your details below to predict your medical insurance charges."
)

st.divider()

# User inputs
st.subheader("Personal Information")

age = st.slider("Select your age", 18, 100, 25)

bmi = st.slider(
    "Select your BMI",
    15.0,
    50.0,
    25.0,
    step=0.1
)

children = st.slider(
    "Number of children",
    0,
    5,
    0
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

smoker = st.selectbox(
    "Do you smoke?",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

st.divider()


if st.button("Predict Insurance Cost"):

    # Convert categorical values
    sex = 0 if gender == "Male" else 1

    smoker_value = 1 if smoker == "Yes" else 0

    region_southeast = 1 if region == "southeast" else 0

    bmi_category_obese = 1 if bmi > 29.9 else 0

    # Create input dataframe
    input_data = pd.DataFrame([{
        "smoker": smoker_value,
        "region_southeast": region_southeast,
        "sex": sex,
        "bmi_category_obese": bmi_category_obese,
        "age": age,
        "bmi": bmi,
        "children": children
    }])
    input_data = input_data[columns]
    numerical_cols = ["age", "bmi", "children"]

    input_data[numerical_cols] = scaler.transform(
        input_data[numerical_cols]
    )


    prediction = model.predict(input_data)

    # Display result
    st.success("Prediction Completed!")

    st.subheader("Estimated Insurance Cost")

    st.title(f"₹{predicted_cost:,.2f}")

    st.write(
        f"Estimated monthly equivalent: ₹{predicted_cost / 12:,.2f}"
    )

    st.info(
        "This is an educational ML prediction using a US insurance dataset. "
        "It is not an actual Indian insurance quotation."
    )

st.divider()

st.caption("InsureAI | Built using Python, Streamlit and Machine Learning")

