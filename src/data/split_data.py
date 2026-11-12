import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw_data/raw.csv"
PROCESSED_DIR = "data/processed_data"

def main():
    df = pd.read_csv(RAW_PATH)

    # Supprimer la colonne date
    df = df.drop(columns=["date"])

    # Variable cible : dernière colonne
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    # Split entraînement / test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

if __name__ == "__main__":
    main()
