import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# === Load preprocessed data ===
print("Loading training data...")
df_train = pd.read_csv('data/train_data.csv')

# Split features and target
X_train = df_train.drop('quality', axis=1)
y_train = df_train['quality']

# === Train model ===
print("Training Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=2000)
model.fit(X_train, y_train)

# === Evaluate on training set (optional check) ===
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_acc:.4f}")

# === Save model ===
os.makedirs('models/train', exist_ok=True)
model_path = 'models/train/model.joblib'
joblib.dump(model, model_path)
print(f"✅ Model saved to '{model_path}'")
