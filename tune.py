import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os


# === Load data ===
print("Loading training data...")
df = pd.read_csv('data/train_data.csv')

X = df.drop('quality', axis=1)
y = df['quality']

# Split for internal validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# === Define parameter grid ===
n_estimators_list = [50, 100]
max_depth_list = [5, 10]

# === Initialize MLflow ===
mlflow.set_experiment("Wine_Quality_Tuning")

with mlflow.start_run(run_name="RandomForest_Tuning") as parent_run:
    print("\nStarting parent MLflow run...")

    for n_est in n_estimators_list:
        for depth in max_depth_list:
            with mlflow.start_run(run_name=f"RF_n{n_est}_d{depth}", nested=True):
                print(f"Training with n_estimators={n_est}, max_depth={depth}")

                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Evaluate model
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                print(f"Validation Accuracy: {acc:.4f}")

                # Log parameters and metrics
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", depth)
                mlflow.log_metric("accuracy", acc)

                # Log model to MLflow and save locally
                mlflow.sklearn.log_model(model, "random_forest_model")
                os.makedirs("models", exist_ok=True)
                model_path = f"models/rf_model_n{n_est}_d{depth}.joblib"
                joblib.dump(model, model_path)

print("\n✅ Tuning complete. Models and logs saved.")
