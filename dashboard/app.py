
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "dashboard" / "student_risk_model_v2.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "student_performance_cleaned.csv"

# Page setup
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df

model = load_model()
df = load_data()

# Header
st.title("🎓 Student Performance Analytics")
st.write("AI-Powered Academic Performance & Risk Detection System")
st.divider()

# Navigation
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard Overview",
    "🔍 Student Risk Prediction",
    "📈 Academic Insights"
])

# TAB 1: OVERVIEW
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", len(df))
    c2.metric("Average Final Grade", round(df["G3"].mean(), 2))
    c3.metric("High Risk Students", int((df["risk_level"] == "High").sum()))
    c4.metric("Low Risk Students", int((df["risk_level"] == "Low").sum()))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Student Risk Distribution")
        risk_counts = df["risk_level"].value_counts()
        st.bar_chart(risk_counts)

    with right:
        st.subheader("Final Grade Distribution")
        grade_counts = df["G3"].value_counts().sort_index()
        st.bar_chart(grade_counts)

    st.subheader("Student Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# TAB 2: PREDICTION
with tab2:
    st.subheader("Predict Individual Student Risk")
    st.write("Enter student information to estimate academic risk.")

    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["F", "M"])
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address", ["U", "R"])

        with c2:
            studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
            failures = st.selectbox("Past Failures", [0, 1, 2, 3])
            absentees = st.number_input(
                "Absences", min_value=0, max_value=100, value=2
            )

        with c3:
            G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
            G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
            health = st.slider("Health Status", 1, 5, 3)

        submitted = st.form_submit_button(
            "Predict Student Risk",
            type="primary"
        )


    if submitted:
        
        input_data = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "address": address,
            "G1": G1,
            "G2": G2,
            "health": health,
            "studytime": studytime,
            "failures": failures,
            "absentees": absentees
        }])


        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == "High":
            st.error("⚠️ High Risk: Academic support may be needed.")
        elif prediction == "Medium":
            st.warning("🟡 Medium Risk: Student performance should be monitored.")
        else:
            st.success("🟢 Low Risk: Student is currently in the low-risk category.")
        st.info(f"Predicted Category: {prediction}")

# TAB 3: INSIGHTS
with tab3:
    st.subheader("Academic Performance Insights")

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Average Grade by Study Time**")
        study_analysis = df.groupby("studytime")["G3"].mean()
        st.bar_chart(study_analysis)

    with c2:
        st.write("**Average Grade by Past Failures**")
        failure_analysis = df.groupby("failures")["G3"].mean()
        st.bar_chart(failure_analysis)

    st.write("**Absences vs Final Grade**")
    st.scatter_chart(df, x="absentees", y="G3")

    st.write("**Risk Category Summary**")
    st.dataframe(
        df["risk_level"].value_counts().rename_axis("Risk Level").reset_index(name="Students"),
        use_container_width=True
    )

st.divider()
st.caption(
    "Student Performance Analytics | Machine Learning | Python | Streamlit"
)