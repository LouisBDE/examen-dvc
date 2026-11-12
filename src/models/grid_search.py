import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

PROCESSED_DIR = "data/processed_data"
MODELS_DIR = "models"

def main():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

    # Modèle choisi : RandomForestRegressor
    # robuste, performant sur des données tabulaires ; insensible à la normalisation
    # (découpe sur des seuils, pas de calcul de distances), mais l'énoncé l'exige.
    model = RandomForestRegressor(random_state=42)

    # Grille réduite : 2x2x1x2 = 8 combinaisons, x 5 folds = 40 fits
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, None],
        "min_samples_split": [5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    print(f"Meilleur score (MSE négatif, CV) : {grid_search.best_score_:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "best_params.pkl"), "wb") as f:
        pickle.dump(grid_search.best_params_, f)

    print(f"Meilleurs paramètres sauvegardés dans {MODELS_DIR}/best_params.pkl")

if __name__ == "__main__":
    main()
