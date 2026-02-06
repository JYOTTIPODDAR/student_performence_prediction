import streamlit as st
import pandas as pd
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction App")
st.write(
    "Predict whether a student is likely to **PASS** or **FAIL** "
    "based on academic and lifestyle factors."
)

# ---------------- LOAD MODEL & COLUMNS ----------------
@st.cache_resource
def load_artifacts():
    with open("model/student_performance_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/columns.pkl", "rb") as f:
        columns = pickle.load(f)

    return model, columns


model, feature_columns = load_artifacts()

# ---------------- INPUT UI ----------------
st.header("📝 Enter Student Details")

# Numeric inputs
hours_studied = st.number_input("Hours Studied", 0.0, 24.0, 7.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0)
sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
previous_scores = st.number_input("Previous Scores", 0.0, 100.0, 90.0)
tutoring_sessions = st.number_input("Tutoring Sessions", 0, 20, 3)
physical_activity = st.number_input("Physical Activity (hrs/week)", 0.0, 20.0, 1.0)

# Categorical dropdowns
teacher_quality = st.selectbox("Teacher Quality", ["High", "Medium", "Low"])
school_type = st.selectbox("School Type", ["Private", "Public"])
peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
motivation = st.selectbox("Motivation Level", ["High", "Medium", "Low"])
parental_involvement = st.selectbox("Parental Involvement", ["High", "Medium", "Low"])
access_resources = st.selectbox("Access to Resources", ["High", "Medium", "Low"])
family_income = st.selectbox("Family Income", ["High", "Medium", "Low"])
parent_edu = st.selectbox(
    "Parental Education Level",
    ["High School", "Undergraduate", "Postgraduate"]
)
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
learning_disability = st.selectbox("Learning Disability", ["No", "Yes"])
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

# ---------------- BUILD INPUT DATA ----------------
input_data = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions,
    "Physical_Activity": physical_activity,
}

# One-hot encoding manually (same as training)
def one_hot(name, value, options):
    return {f"{name}_{opt}": 1 if value == opt else 0 for opt in options}


input_data.update(one_hot("Teacher_Quality", teacher_quality, ["Low", "Medium", "High"]))
input_data.update(one_hot("School_Type", school_type, ["Public", "Private"]))
input_data.update(one_hot("Peer_Influence", peer_influence, ["Negative", "Neutral", "Positive"]))
input_data.update(one_hot("Motivation_Level", motivation, ["Low", "Medium", "High"]))
input_data.update(one_hot("Parental_Involvement", parental_involvement, ["Low", "Medium", "High"]))
input_data.update(one_hot("Access_to_Resources", access_resources, ["Low", "Medium", "High"]))
input_data.update(one_hot("Family_Income", family_income, ["Low", "Medium", "High"]))
input_data.update(one_hot(
    "Parental_Education_Level",
    parent_edu,
    ["High School", "Undergraduate", "Postgraduate"]
))

input_data["Internet_Access_Yes"] = 1 if internet_access == "Yes" else 0
input_data["Learning_Disabilities_Yes"] = 1 if learning_disability == "Yes" else 0
input_data["Extracurricular_Activities_Yes"] = 1 if extracurricular == "Yes" else 0

# Create DataFrame
input_df = pd.DataFrame([input_data])

# Ensure column order matches training
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("🎯 Predict Result"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Student is likely to **PASS**")
    else:
        st.error("❌ Student is likely to **FAIL**")

    # Debug (optional)
    with st.expander("🔍 Debug: Model Input"):
        st.dataframe(input_df)
