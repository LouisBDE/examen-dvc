import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = "data/processed_data"
MODELS_DIR = "models"

def main():
    # Récupération des meilleurs paramètres issus de grid_search.py
    with open(os.path.join(MODELS_DIR, "best_params.pkl"), "rb") as f:
        best_params = pickle.load(f)

    print(f"Paramètres chargés depuis {MODELS_DIR}/best_params.pkl : {best_params}")

    # Entraînement : uniquement les données de train, rien de plus
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

    model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)

    # Vérification des paramètres effectifs du modèle (dont random_state)
    print(f"Paramètres du modèle : {model.get_params()}")

    model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    with open(os.path.join(MODELS_DIR, "random_forest_trained.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"Modèle final sauvegardé dans {MODELS_DIR}/random_forest_trained.pkl")

if __name__ == "__main__":
    main()
