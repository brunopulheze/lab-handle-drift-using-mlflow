import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evidently import Report
from evidently.presets import DataDriftPreset

# Load data and prep
iris_data = load_iris(as_frame=True)
df = iris_data.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 1: Train and log baseline model using MLflow autologging
mlflow.autolog()

with mlflow.start_run(run_name="baseline_random_forest"):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

# Step 2: Simulate feature drift
X_drifted = X_test.copy()
X_drifted["sepal length (cm)"] += np.random.normal(loc=2.0, scale=0.3, size=len(X_drifted))

with mlflow.start_run(run_name="drifted_random_forest"):
    model_drifted = RandomForestClassifier()
    model_drifted.fit(X_train, y_train)

    # Step 3: Detect drift using Evidently
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=X_train, current_data=X_drifted)
    result.save_html("drift_report.html")

    # Step 4: Log the drift report as an artifact in MLflow
    mlflow.log_artifact("drift_report.html")