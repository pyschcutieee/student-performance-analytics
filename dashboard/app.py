
import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# ---------------- FILE PATHS ----------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "dashboard" / "student_risk_model_v2.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "student_performance_cleaned.csv"

sys.path.insert(0, str(BASE_DIR))

from src.recommendation_engine import (
    generate_recommendations,
    generate_study_plan
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)


# ---------------- LOAD MODEL & DATA ----------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    return data


def get_expected_features(trained_model):

    if hasattr(trained_model, "feature_names_in_"):
        return list(trained_model.feature_names_in_)

    if hasattr(trained_model, "named_steps"):
        for step in trained_model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    raise ValueError(
        "Model feature names are unavailable."
    )


try:
    model = load_model()
    df = load_data()

except Exception as e:
    st.error("Failed to load model or dataset.")
    st.exception(e)
    st.stop()


required_columns = {
    "G3",
    "risk_level",
    "studytime",
    "failures",
    "absentees"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        f"Missing dataset columns: {sorted(missing_columns)}"
    )
    st.stop()


# ---------------- HEADER ----------------

st.title("🎓 Student Performance Analytics")

st.write(
    "AI-Powered Academic Performance & Risk Detection System"
)

st.caption(
    "Academic risk predictions use first-period and "
    "second-period grades, after G2 results are available."
)

st.divider()


# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard Overview",
    "🔍 Student Risk Prediction",
    "📈 Academic Insights"
])


# ==================================================
# TAB 1 - DASHBOARD OVERVIEW
# ==================================================

with tab1:

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        len(df)
    )

    c2.metric(
        "Average Final Grade",
        f"{df['G3'].mean():.2f} / 20"
    )

    c3.metric(
        "High Risk Students",
        int((df["risk_level"] == "High").sum())
    )

    c4.metric(
        "Low Risk Students",
        int((df["risk_level"] == "Low").sum())
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Student Risk Distribution")

        risk_counts = (
            df["risk_level"]
            .value_counts()
            .reindex(
                ["High", "Medium", "Low"],
                fill_value=0
            )
        )

        st.bar_chart(risk_counts)

    with right:

        st.subheader("Final Grade Distribution")

        grade_counts = (
            df["G3"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(grade_counts)

    st.subheader("Student Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==================================================
# TAB 2 - STUDENT RISK PREDICTION
# ==================================================

with tab2:

    st.subheader("Predict Individual Student Risk")

    st.write(
        "Enter student details to estimate academic risk."
    )

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        with c1:

            gender = st.selectbox(
                "Gender",
                ["F", "M"]
            )

            age = st.slider(
                "Age",
                15,
                22,
                17
            )

            address = st.selectbox(
                "Address Type",
                ["U", "R"]
            )

        with c2:

            studytime = st.selectbox(
                "Weekly Study Time",
                [1, 2, 3, 4]
            )

            failures = st.selectbox(
                "Past Failures",
                [0, 1, 2, 3]
            )

            absentees = st.number_input(
                "Absentees",
                min_value=0,
                max_value=100,
                value=2,
                step=1
            )

        with c3:

            G1 = st.slider(
                "First Period Grade (G1)",
                0,
                20,
                10
            )

            G2 = st.slider(
                "Second Period Grade (G2)",
                0,
                20,
                10
            )

            health = st.slider(
                "Health Status",
                1,
                5,
                3
            )

        submitted = st.form_submit_button(
            "Predict Student Risk",
            type="primary"
        )

    if submitted:

        try:

            # Student input features

            student = {
                "gender": gender,
                "age": age,
                "address": address,
                "G1": G1,
                "G2": G2,
                "health": health,
                "studytime": studytime,
                "failures": failures,
                "absentees": int(absentees)
            }

            # Validate model features

            expected_features = get_expected_features(model)

            missing_features = [
                feature
                for feature in expected_features
                if feature not in student
            ]

            extra_features = [
                feature
                for feature in student
                if feature not in expected_features
            ]

            if missing_features or extra_features:

                raise ValueError(
                    f"Missing features: {missing_features}. "
                    f"Unexpected features: {extra_features}."
                )

            input_data = pd.DataFrame(
                [student],
                columns=expected_features
            )

            # Model prediction

            prediction = str(
                model.predict(input_data)[0]
            ).strip()

            st.divider()

            st.subheader("Prediction Result")

            if prediction in ["High", "High Risk"]:

                st.error(
                    "🔴 HIGH RISK: Student may require "
                    "additional academic support."
                )

            elif prediction in ["Medium", "Medium Risk"]:

                st.warning(
                    "🟡 MEDIUM RISK: Student performance "
                    "should be monitored."
                )

            elif prediction in ["Low", "Low Risk"]:

                st.success(
                    "🟢 LOW RISK: Student is currently "
                    "in the low-risk category."
                )

            else:

                st.warning(
                    f"Unknown prediction category: {prediction}"
                )

            st.info(
                f"Predicted Risk Category: {prediction}"
            )

            # ========================================
            # PERSONALIZED ACADEMIC RECOMMENDATIONS
            # ========================================

            st.divider()

            st.subheader(
                "🎯 Personalized Academic Recommendations"
            )

            recommendations = generate_recommendations(
                student,
                prediction
            )

            for i, recommendation in enumerate(
                recommendations,
                1
            ):
                st.write(
                    f"{i}. {recommendation}"
                )

            # ========================================
            # SMART DAILY STUDY PLANNER
            # ========================================

            st.divider()

            st.subheader(
                "📅 Smart Daily Study Planner"
            )

            study_plan = generate_study_plan(
                student,
                prediction
            )

            plan_df = pd.DataFrame(
                study_plan,
                columns=[
                    "Time Slot",
                    "Recommended Activity"
                ]
            )

            st.dataframe(
                plan_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "This is a suggested study routine. "
                "Actual academic improvement depends on "
                "consistent practice, attendance, "
                "and individual learning needs."
            )

            # ========================================
            # PREDICTION PROBABILITIES
            # ========================================

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                classes = getattr(
                    model,
                    "classes_",
                    None
                )

                if (
                    classes is None
                    and hasattr(model, "named_steps")
                ):

                    for step in reversed(
                        list(model.named_steps.values())
                    ):

                        if hasattr(step, "classes_"):
                            classes = step.classes_
                            break

                if classes is not None:

                    probability_df = pd.DataFrame({
                        "Risk Category": classes,
                        "Probability": probabilities
                    }).set_index("Risk Category")

                    st.divider()

                    st.subheader(
                        "Prediction Probability Distribution"
                    )

                    st.bar_chart(
                        probability_df
                    )

                    st.caption(
                        "Probabilities are model estimates "
                        "and may not be perfectly calibrated."
                    )

        except Exception as e:

            st.error(
                "Prediction failed. Please check "
                "the model features and configuration."
            )

            st.exception(e)


# ==================================================
# TAB 3 - ACADEMIC INSIGHTS
# ==================================================

with tab3:

    st.subheader(
        "Academic Performance Insights"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            "**Average Grade by Study Time**"
        )

        study_analysis = (
            df.groupby("studytime")["G3"]
            .mean()
        )

        st.bar_chart(
            study_analysis
        )

    with c2:

        st.write(
            "**Average Grade by Past Failures**"
        )

        failure_analysis = (
            df.groupby("failures")["G3"]
            .mean()
        )

        st.bar_chart(
            failure_analysis
        )

    st.write(
        "**Absentees vs Final Grade**"
    )

    st.scatter_chart(
        df,
        x="absentees",
        y="G3"
    )

    st.write(
        "**Risk Category Summary**"
    )

    risk_summary = (
        df["risk_level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Students")
    )

    st.dataframe(
        risk_summary,
        use_container_width=True
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Student Performance Analytics | "
    "Machine Learning | Python | Streamlit | "
    "Smart Study Planner"
)