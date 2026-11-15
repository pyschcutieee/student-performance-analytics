import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "dashboard" / "student_risk_model_v2.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "student_performance_cleaned.csv"

# ---------------- PAGE SETUP ----------------

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

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error("Model or dataset could not be loaded.")
    st.exception(e)
    st.stop()

required_columns = {
    "G3", "risk_level", "studytime", "failures", "absentees"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(f"Missing dataset columns: {sorted(missing_columns)}")
    st.stop()

# ---------------- HEADER ----------------

st.title("🎓 Student Performance Analytics")

st.write(
    "AI-Powered Academic Performance & Risk Detection System"
)

st.caption(
    "Predictions use G1 and G2 grades, so they apply "
    "after second-period grades are available."
)

st.divider()

# ---------------- NAVIGATION ----------------

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard Overview",
    "🔍 Student Risk Prediction",
    "📈 Academic Insights"
])

# ================= TAB 1 =================

with tab1:

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", len(df))

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
            .reindex(["High", "Medium", "Low"], fill_value=0)
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

# ================= TAB 2 =================

with tab2:

    st.subheader("Predict Individual Student Risk")

    st.write(
        "Enter student information to estimate academic risk."
    )

    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["F", "M"])
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address", ["U", "R"])

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
                "Absences",
                min_value=0,
                max_value=100,
                value=2,
                step=1
            )

        with c3:
            G1 = st.slider(
                "First Period Grade (G1)",
                0, 20, 10
            )

            G2 = st.slider(
                "Second Period Grade (G2)",
                0, 20, 10
            )

            health = st.slider(
                "Health Status",
                1, 5, 3
            )

        submitted = st.form_submit_button(
            "Predict Student Risk",
            type="primary"
        )

    if submitted:

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

        try:
            # Get exact feature order used during training

            if hasattr(model, "feature_names_in_"):
                expected_features = list(
                    model.feature_names_in_
                )

            elif (
                hasattr(model, "named_steps")
                and "preprocessor" in model.named_steps
                and hasattr(
                    model.named_steps["preprocessor"],
                    "feature_names_in_"
                )
            ):
                expected_features = list(
                    model.named_steps[
                        "preprocessor"
                    ].feature_names_in_
                )

            else:
                raise ValueError(
                    "Training feature names not found in model."
                )

            # Validate feature names

            missing_features = [
                col for col in expected_features
                if col not in student
            ]

            extra_features = [
                col for col in student
                if col not in expected_features
            ]

            if missing_features or extra_features:
                raise ValueError(
                    f"Missing features: {missing_features}; "
                    f"Unexpected features: {extra_features}"
                )

            # IMPORTANT FIX:
            # Match model's exact training feature order

            input_data = pd.DataFrame(
                [student],
                columns=expected_features
            )

            prediction = str(
                model.predict(input_data)[0]
            )

            st.divider()

            st.subheader("Prediction Result")

            if prediction == "High":
                st.error(
                    "⚠️ High Risk: Academic support may be needed."
                )

            elif prediction == "Medium":
                st.warning(
                    "🟡 Medium Risk: Student performance "
                    "should be monitored."
                )

            elif prediction == "Low":
                st.success(
                    "🟢 Low Risk: Student is currently "
                    "in the low-risk category."
                )

            else:
                st.warning(
                    f"Unexpected prediction: {prediction}"
                )

            st.info(
                f"Predicted Category: {prediction}"
            )

            # Show model probabilities

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                if hasattr(model, "classes_"):
                    classes = model.classes_
                else:
                    classes = model.named_steps[
                        "classifier"
                    ].classes_

                probability_df = pd.DataFrame({
                    "Risk Category": classes,
                    "Model Probability": probabilities
                }).set_index("Risk Category")

                st.subheader(
                    "Model Probability by Category"
                )

                st.bar_chart(probability_df)

                st.caption(
                    "These are model-estimated probabilities, "
                    "not guaranteed real-world confidence."
                )

        except Exception as e:
            st.error(
                "Prediction failed. Please check model "
                "features and dependencies."
            )
            st.exception(e)

# ================= TAB 3 =================

with tab3:

    st.subheader("Academic Performance Insights")

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Average Grade by Study Time**")

        study_analysis = (
            df.groupby("studytime")["G3"].mean()
        )

        st.bar_chart(study_analysis)

    with c2:
        st.write("**Average Grade by Past Failures**")

        failure_analysis = (
            df.groupby("failures")["G3"].mean()
        )

        st.bar_chart(failure_analysis)

    st.write("**Absences vs Final Grade**")

    st.scatter_chart(
        df,
        x="absentees",
        y="G3"
    )

    st.write("**Risk Category Summary**")

    summary = (
        df["risk_level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Students")
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Student Performance Analytics | "
    "Machine Learning | Python | Streamlit"
)