import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# === Load raw data ===
print("Loading raw data...")
df = pd.read_csv('data/wine_data_raw.csv')

def preprocess_data(df):
    """Perform basic cleaning and validation."""
    print("\nInitial shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum())

    # Drop rows with missing values
    df = df.dropna()
    print("Shape after removing NaN:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()
    print("Shape after removing duplicates:", df.shape)

    # Ensure target column exists
    if 'quality' not in df.columns:
        raise ValueError("Target column 'quality' not found in dataset.")

    return df

# === Clean data ===
print("\nPreprocessing data...")
df_clean = preprocess_data(df)

# === Split train/test ===
print("\nSplitting dataset...")
X = df_clean.drop('quality', axis=1)
y = df_clean['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Save processed files ===
os.makedirs('data', exist_ok=True)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

print("\n✅ Preprocessing complete.")
print(f"Training set shape: {train_data.shape}")
print(f"Test set shape: {test_data.shape}")
