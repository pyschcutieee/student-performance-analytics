import pandas as pd

df = pd.read_csv("data/raw/student_performance_cleaned.csv")

print("\n===== STUDENT PERFORMANCE ANALYSIS =====")

print("\nTotal Students:", len(df))
print("Total Columns:", len(df.columns))

print("\nAverage Grades:")
print(df[["G1", "G2", "G3"]].mean().round(2))

print("\nRisk Level Distribution:")
print(df["risk_level"].value_counts())

print("\nGender-wise Performance:")
print(df.groupby("gender")["G3"].mean().round(2))

print("\nStudy Time vs Performance:")
print(df.groupby("studytime")["G3"].mean().round(2))

print("\nAbsences and Grade Correlation:")
print(df["absentees"].corr(df["G3"]))

summary = df.groupby("risk_level").agg(
    total_students=("name", "count"),
    average_grade=("G3", "mean"),
    average_absences=("absentees", "mean")
).round(2)

summary.to_csv("data/risk_summary.csv")

print("\nRisk Summary:")
print(summary)

print("\nAnalysis completed successfully!")
