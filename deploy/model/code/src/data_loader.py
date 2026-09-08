import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(filepath: str, text_col: str = "text", label_col: str = "label"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV not found at: {filepath}")

    print(f"[DataLoader] Loading CSV from: {filepath}")
    df = pd.read_csv(filepath)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns. "
                         f"Found: {list(df.columns)}")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df.dropna(inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[DataLoader] Loaded {len(df)} samples.")
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
    X = df["text"].to_numpy()
    y = df["label"].to_numpy()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val, random_state=42, stratify=y_temp
    )

    print(f"[DataLoader] Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test