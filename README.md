# Student Performance Analytics & Early Academic Risk Detection

An end-to-end Data Analytics and Machine Learning project that analyzes student academic performance, identifies performance patterns, and predicts student risk categories using machine learning.

## Project Objectives

- Analyze student academic performance.
- Identify factors associated with final grades.
- Perform exploratory data analysis and statistical analysis.
- Group students using clustering techniques.
- Predict academic risk using machine learning.
- Visualize insights through an interactive Streamlit dashboard.

## Technologies Used

- **Programming:** Python, R
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Streamlit
- **Machine Learning:** Scikit-learn
- **Statistical Analysis:** R, SciPy
- **Development:** Jupyter Notebook, VS Code, Git, GitHub

## Dataset

UCI Student Performance Dataset.

The project uses the Mathematics student dataset (`student-mat.csv`).

## Project Structure

```text
student-performance-analytics/
│
├── dashboard/
│   ├── app.py
│   └── student_risk_model.pkl
│
├── data/
│   ├── raw/
│   │   └── student-mat.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_statistical_analysis.ipynb
│   └── 03_performance_prediction.ipynb
│
├── reports/
│
├── src/
│   ├── check_dataset.py
│   └── statistical_analysis.R
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Machine Learning Models

The project compares:

1. Logistic Regression
2. Random Forest Classifier

The models classify students into three categories:

- **High Risk:** Final grade below 10
- **Medium Risk:** Final grade from 10 to 14
- **Low Risk:** Final grade 15 or above

Model performance is evaluated using accuracy, macro F1-score, classification reports, and a confusion matrix.

Note: The current prediction model uses G1 and G2 grades, so it estimates final academic risk after the second-period grades are available.

## Dashboard Features

- Total student count
- Average final grade
- Student risk distribution
- Grade distribution
- Individual student risk prediction
- Study-time and academic performance analysis
- Absences versus final grade visualization

## How to Run the Project

### 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd student-performance-analytics
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment (Windows)

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Project Outcome

An interactive academic analytics system combining data analysis, statistical techniques, machine learning, and visualization to support student performance monitoring and risk assessment.

## Disclaimer

This project is developed for educational and analytical purposes. Predictions are estimates based on historical dataset patterns and should not be treated as definitive judgments about individual students.