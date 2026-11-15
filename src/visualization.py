
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv("data/raw/student_performance_cleaned.csv")

# Create output folder
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# Graph 1: Academic Risk Distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="risk_level",
    order=["Low", "Medium", "High"],
    hue="risk_level",
    palette="Set2",
    legend=False
)
plt.title("Academic Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(output_dir / "risk_distribution.png")
plt.close()

# Graph 2: Study Time vs Final Grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(output_dir / "studytime_vs_grade.png")
plt.close()

# Graph 3: Absences vs Final Grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absentees", y="G3")
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(output_dir / "absences_vs_grade.png")
plt.close()

# Graph 4: Grade Correlation Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(
    df[["G1", "G2", "G3"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Grade Correlation Heatmap")
plt.tight_layout()
plt.savefig(output_dir / "grade_correlation.png")
plt.close()

print("All 4 graphs generated successfully!")