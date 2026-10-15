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

    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=DROP_COLUMNS)

    train_df, test_df = train_test_split(df,
                                         test_size=params["test_size"],
                                         random_state=params["random_state"],
                                         stratify=df[TARGET],
                                         )

    os.makedirs("data/processed", exist_ok=True)
    
    taux_train = train_df[TARGET].mean()
    taux_test = test_df[TARGET].mean()

    train_csv = train_df.to_csv(TRAIN_PATH, index=False)
    test_csv = test_df.to_csv(TEST_PATH, index=False)

    print(f"nombre de ligne du train : {len(train_df)}, taux : {taux_train}")
    print(f"nombre de ligne du test : {len(test_df)}, taux : {taux_test}")

    