import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

CSV_PATH = "data/raw/data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100 
MAX_DEPTH = None
EXPERIMENT_NAME = "predictive-maintenance"
MODEL_NAME = "predictive-maintenance"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

    X = df.drop(columns = ["Machine failure"])
    y = df["Machine failure"]

    return X, y


def split(X, y): 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def build_pipeline():
    # TODO : créer le ColumnTransformer.
    #   Il prend une liste de tuples (nom, transformateur, colonnes).
    #   Ici un seul tuple : nom "type", transformateur OneHotEncoder(), colonnes ["Type"].
    #   Ajoute remainder="passthrough" pour que les 5 colonnes numériques passent telles quelles.
    t = [("type", OneHotEncoder(), ["Type"])]
    preprocessing = ColumnTransformer(transformers = t, remainder="passthrough")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )

    # TODO : créer le Pipeline.
        #   Il prend une liste de tuples (nom, étape), dans l'ordre d'exécution :
        #   ("preprocessing", preprocessing) puis ("model", RandomForestClassifier(...)).
        #   Passe N_ESTIMATORS, MAX_DEPTH et RANDOM_STATE au RandomForestClassifier.
    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])

    return pipeline


def evaluation(y_true, y_pred):
    # TODO : calculer le F1, la précision et le rappel avec les fonctions de sklearn.metrics.
    # TODO : les mettre dans un dictionnaire {"f1": ..., "precision": ..., "recall": ...}.
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    metrics = {
        "f1": f1, 
        "precision": precision, 
        "recall": recall
        }
    return metrics


if __name__ == "__main__":
    X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = split(X, y)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluation(y_test, y_pred)
        print(metrics)

        # TODO : logger les 4 hyperparamètres avec mlflow.log_params().
        #   Elle prend un dictionnaire {"nom": valeur} : n_estimators, max_depth, random_state, test_size.
        mlflow.log_params({"n_estimators": N_ESTIMATORS,  
                           "max_depth": MAX_DEPTH, 
                           "random_state": RANDOM_STATE, 
                           "test_size": TEST_SIZE})

        # TODO : logger les métriques avec mlflow.log_metrics() (ton dictionnaire metrics, tel quel).
        mlflow.log_metrics(metrics)

        # TODO : calculer la signature avec infer_signature(entrées, sorties).
        #   Entrées : X_train. Sorties : les prédictions du pipeline sur X_train.
        signature = infer_signature(X_train, pipeline.predict(X_train))

        # TODO : logger et enregistrer le modèle avec mlflow.sklearn.log_model(), qui renvoie un objet ModelInfo.
        #   Paramètres : sk_model=pipeline, name="model", signature=signature,
        #   input_example=X_train.head(5), registered_model_name=MODEL_NAME,
        #   skops_trusted_types=["sklearn.tree._tree.Tree"]
        model_info = mlflow.sklearn.log_model(sk_model=pipeline, 
                                 name="model", 
                                 signature=signature, 
                                 input_example=X_train.head(5), 
                                 registered_model_name=MODEL_NAME,
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

    # TODO : poser l'alias "champion" avec MlflowClient().set_registered_model_alias().
    #   Arguments : le nom du modèle, "champion", et le numéro de version,
    #   disponible dans model_info.registered_model_version.
    version = model_info.registered_model_version
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version)