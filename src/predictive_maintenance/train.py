import json
import os

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predictive_maintenance.common import (
    CANDIDATE_PATH,
    EXPERIMENT_NAME,
    MODEL_NAME,
    PARAMS_PATH,
    TRAIN_PATH,
    load_params,
    split_xy,
)


def build_pipeline(n_estimators, max_depth, class_weight, random_state):
    t = [("type", OneHotEncoder(), ["Type"])]
    preprocessing = ColumnTransformer(transformers = t, remainder="passthrough")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
    )
    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])

    return pipeline


if __name__ == "__main__":
    params = load_params(PARAMS_PATH)["train"]

    train_df = pd.read_csv(TRAIN_PATH)
    X_train, y_train = split_xy(train_df)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        pipeline = build_pipeline(params["n_estimators"], 
                                          params["max_depth"], 
                                          params["class_weight"], 
                                          params["random_state"])
        pipeline.fit(X_train, y_train)

        mlflow.log_params(params)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        
        model_info = mlflow.sklearn.log_model(sk_model=pipeline, 
                                              name="model", 
                                              signature=signature, 
                                              input_example=X_train.head(5), 
                                              registered_model_name=MODEL_NAME,
                                              skops_trusted_types=["sklearn.tree._tree.Tree"])

    os.makedirs("outputs", exist_ok=True)

    dict = {"run_id": run.info.run_id, 
            "version": model_info.registered_model_version}
    
    with open(CANDIDATE_PATH, "w", encoding="utf-8") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)