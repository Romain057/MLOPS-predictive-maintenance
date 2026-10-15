import os

import pandas as pd
from sklearn.model_selection import train_test_split

from predictive_maintenance.common import (
    DROP_COLUMNS,
    PARAMS_PATH,
    RAW_PATH,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    load_params,
    
)

if __name__ == "__main__":
    params = load_params(PARAMS_PATH)["prepare"]

    # TODO : lire RAW_PATH et retirer les colonnes DROP_COLUMNS.
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=DROP_COLUMNS)

    # TODO : découper df (le DataFrame entier, pas X et y) en train_df et test_df avec train_test_split.
    #   test_size et random_state viennent de params, stratify sur la colonne df[TARGET].
    train_df, test_df = train_test_split(df,
                                         test_size=params["test_size"],
                                         random_state=params["random_state"],
                                         stratify=df[TARGET],
                                         )

    # TODO : créer le dossier data/processed s'il n'existe pas : os.makedirs(..., exist_ok=True).
    os.makedirs("data/processed", exist_ok=True)
    # TODO : écrire train_df dans TRAIN_PATH et test_df dans TEST_PATH, avec index=False.
    taux_train = train_df[TARGET].mean()
    taux_test = test_df[TARGET].mean()

    train_csv = train_df.to_csv(TRAIN_PATH, index=False)
    test_csv = test_df.to_csv(TEST_PATH, index=False)
    # TODO : afficher le nombre de lignes et le taux de panne de chaque fichier, pour contrôle.
    print(f"nombre de ligne du train : {len(train_df)}, taux : {taux_train}")
    print(f"nombre de ligne du test : {len(test_df)}, taux : {taux_test}")

    