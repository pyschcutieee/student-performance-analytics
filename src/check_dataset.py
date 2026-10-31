import pandas as pd

# Load student mathematics dataset
data = pd.read_csv("data/raw/student-mat.csv", sep=";")

# Display basic information
print("STUDENT PERFORMANCE DATASET")
print("----------------------------")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns.tolist())

print("\nDataset Information:")
print(data.info())


print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nStatistical Summary:")
print(data.describe())

print("\nTarget Grade Distribution:")
print(data["G3"].value_counts().sort_index())
