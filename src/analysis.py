
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/raw/student_performance_cleaned.csv")

print("\n===== STUDENT PERFORMANCE ANALYSIS =====")

# Dataset information
print("\nTotal Students:", len(df))
print("Total Columns:", len(df.columns))

# Average grades
print("\nAverage Grades:")
print(df[["G1", "G2", "G3"]].mean().round(2))

# Academic risk distribution
print("\nRisk Level Distribution:")
print(df["risk_level"].value_counts())

# Gender-wise average final grades
print("\nGender-wise Performance:")
print(df.groupby("gender")["G3"].mean().round(2))

# Study time vs final grades
print("\nStudy Time vs Performance:")
print(df.groupby("studytime")["G3"].mean().round(2))

# Absences vs final grades
print("\nAbsences and Grade Correlation:")
print(df["absentees"].corr(df["G3"]))

# Save summary
summary = df.groupby("risk_level").agg(
    total_students=("name", "count"),
    average_grade=("G3", "mean"),
    average_absences=("absentees", "mean")
).round(2)

summary.to_csv("data/risk_summary.csv")

print("\nRisk Summary:")
print(summary)

print("\nAnalysis completed successfully!")
