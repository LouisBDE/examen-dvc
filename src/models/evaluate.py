import os
import json
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

PROCESSED_DIR = "data/processed_data"
MODELS_DIR = "models"
DATA_DIR = "data"
METRICS_DIR = "metrics"

def main():
    # Chargement du modèle entraîné par train_model.py
    with open(os.path.join(MODELS_DIR, "random_forest_trained.pkl"), "rb") as f:
        model = pickle.load(f)

    # Données de test uniquement
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

    # Prédictions
    y_pred = model.predict(X_test)

    # Dataset des prédictions : X_test + valeur réelle + valeur prédite
    predictions = X_test.copy()
    predictions["silica_concentrate_real"] = y_test
    predictions["silica_concentrate_pred"] = y_pred
    predictions.to_csv(os.path.join(DATA_DIR, "prediction.csv"), index=False)

    # Métriques d'évaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
    }

    print(f"MSE  : {scores['mse']}")
    print(f"RMSE : {scores['rmse']}")
    print(f"MAE  : {scores['mae']}")
    print(f"R²   : {scores['r2']}")

    # Sauvegarde des métriques en JSON (exploitable par dvc metrics)
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "scores.json"), "w") as f:
        json.dump(scores, f, indent=4)

    print(f"Prédictions sauvegardées dans {DATA_DIR}/prediction.csv")
    print(f"Métriques sauvegardées dans {METRICS_DIR}/scores.json")

if __name__ == "__main__":
    main()
