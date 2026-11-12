import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = "data/processed_data"

def main():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))

    # Normalisation : StandardScaler (moyenne 0, écart-type 1)
    # ajusté uniquement sur les données d'entraînement pour éviter les fuites
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"), index=False)

    print(f"X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")

    # Moyenne de chaque colonne, arrondie à 2 décimales
    print("\nMoyennes X_train_scaled :")
    print(X_train_scaled.mean().round(2))
    print("\nMoyennes X_test_scaled :")
    print(X_test_scaled.mean().round(2))


if __name__ == "__main__":
    main()
