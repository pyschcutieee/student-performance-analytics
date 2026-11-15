import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PATHS AND PAGE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "dashboard" / "student_risk_model_v2.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "student_performance_cleaned.csv"

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# 2. LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    return data


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


# ==========================================
# 3. MODEL FEATURE VALIDATION
# ==========================================

def get_expected_features(trained_model):

    if hasattr(trained_model, "feature_names_in_"):
        return list(trained_model.feature_names_in_)

    if hasattr(trained_model, "named_steps"):

        for step in trained_model.named_steps.values():

            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    raise ValueError(
        "The trained model does not expose its input feature names."
    )


# ==========================================
# 4. HEADER
# ==========================================

st.title("🎓 Student Performance Analytics")

st.write(
    "AI-Powered Academic Performance & Risk Detection System"
)

st.caption(
    "Predictions use first-period and second-period grades. "
    "They are intended for academic risk estimation after "
    "G2 grades are available."
)

st.divider()


# ==========================================
# 5. NAVIGATION
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard Overview",
    "🔍 Student Risk Prediction",
    "📈 Academic Insights"
])


# ==========================================
# TAB 1: DASHBOARD OVERVIEW
# ==========================================

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


# ==========================================
# TAB 2: STUDENT RISK PREDICTION
# ==========================================

with tab2:

    st.subheader("Predict Individual Student Risk")

    st.write(
        "Enter student details to estimate academic risk."
    )

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        # COLUMN 1

        with c1:

            age = st.slider(
                "Age",
                min_value=15,
                max_value=22,
                value=17
            )

            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4]
            )

            failures = st.selectbox(
                "Past Failures",
                options=[0, 1, 2, 3]
            )

        # COLUMN 2

        with c2:

            absentees = st.number_input(
                "Absentees",
                min_value=0,
                max_value=100,
                value=2,
                step=1
            )

            goout = st.slider(
                "Going Out Frequency",
                min_value=1,
                max_value=5,
                value=3
            )

            freetime = st.slider(
                "Free Time",
                min_value=1,
                max_value=5,
                value=3
            )

        # COLUMN 3

        with c3:

            G1 = st.slider(
                "First Period Grade (G1)",
                min_value=0,
                max_value=20,
                value=10
            )

            G2 = st.slider(
                "Second Period Grade (G2)",
                min_value=0,
                max_value=20,
                value=10
            )

            health = st.slider(
                "Health Status",
                min_value=1,
                max_value=5,
                value=3
            )

        submitted = st.form_submit_button(
            "Predict Student Risk",
            type="primary"
        )

    # ======================================
    # PREDICTION LOGIC
    # ======================================

    if submitted:

        try:

            # Match exact features used in training.
            # The model expects "absences".
            # The cleaned dataset uses "absentees".

            student = {
                "age": age,
                "studytime": studytime,
                "failures": failures,
                "absences": int(absentees),
                "G1": G1,
                "G2": G2,
                "goout": goout,
                "health": health,
                "freetime": freetime
            }

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

            # Create DataFrame with exact feature order.

            input_data = pd.DataFrame(
                [student],
                columns=expected_features
            )

            # Predict risk category.

            prediction = str(
                model.predict(input_data)[0]
            )

            st.divider()

            st.subheader("Prediction Result")

            if prediction == "High":

                st.error(
                    "⚠️ HIGH RISK: "
                    "Student may require additional academic support."
                )

            elif prediction == "Medium":

                st.warning(
                    "🟡 MEDIUM RISK: "
                    "Student performance should be monitored."
                )

            elif prediction == "Low":

                st.success(
                    "🟢 LOW RISK: "
                    "Student is currently in the low-risk category."
                )

            else:

                st.warning(
                    f"Unexpected prediction category: {prediction}"
                )

            st.info(
                f"Predicted Risk Category: {prediction}"
            )

            # ==================================
            # PREDICTION PROBABILITIES
            # ==================================

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                if hasattr(model, "classes_"):

                    classes = model.classes_

                else:

                    classes = None

                    if hasattr(model, "named_steps"):

                        for step in reversed(
                            list(model.named_steps.values())
                        ):

                            if hasattr(step, "classes_"):
                                classes = step.classes_
                                break

                    if classes is None:
                        raise ValueError(
                            "Model class labels are unavailable."
                        )

                probability_df = pd.DataFrame({
                    "Risk Category": classes,
                    "Probability": probabilities
                }).set_index("Risk Category")

                st.subheader(
                    "Prediction Probability Distribution"
                )

                st.bar_chart(probability_df)

                st.caption(
                    "Probabilities are model estimates and "
                    "may not be perfectly calibrated."
                )

        except Exception as e:

            st.error(
                "Prediction failed. Please check the "
                "model features and configuration."
            )

            st.exception(e)


# ==========================================
# TAB 3: ACADEMIC INSIGHTS
# ==========================================

with tab3:

    st.subheader("Academic Performance Insights")

    c1, c2 = st.columns(2)

    with c1:

        st.write("**Average Grade by Study Time**")

        study_analysis = (
            df.groupby("studytime")["G3"]
            .mean()
        )

        st.bar_chart(study_analysis)

    with c2:

        st.write("**Average Grade by Past Failures**")

        failure_analysis = (
            df.groupby("failures")["G3"]
            .mean()
        )

        st.bar_chart(failure_analysis)

    st.write("**Absentees vs Final Grade**")

    st.scatter_chart(
        df,
        x="absentees",
        y="G3"
    )

    st.write("**Risk Category Summary**")

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


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Student Performance Analytics | "
    "Machine Learning | Python | Streamlit"
)