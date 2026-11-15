import pandas as pd

df = pd.read_csv("data/raw/student-mat.csv", sep=";")

# Anonymous student identifiers
df.insert(
    0,
    "name",
    [f"Student_{i:03d}" for i in range(1, len(df) + 1)]
)

# Risk classification based on G3
def classify_risk(grade):
    if grade < 10:
        return "High"
    elif grade < 14:
        return "Medium"
    return "Low"

df["risk_level"] = df["G3"].apply(classify_risk)

# Rename columns
df = df.rename(columns={
    "sex": "gender",
    "absences": "absentees"
})

# Keep only selected columns
columns = [
    "name", "gender", "age", "address",
    "G1", "G2", "G3", "risk_level",
    "health", "studytime", "failures", "absentees"
]

df = df[columns]

df.to_csv("student_performance_cleaned.csv", index=False)

print("Dataset cleaned successfully!")
print(df.head())