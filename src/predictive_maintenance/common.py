import yaml
import os
import hashlib
import subprocess

RAW_PATH = "data/raw/data.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
PARAMS_PATH = "params.yaml"
CANDIDATE_PATH = "outputs/candidate.json"
EVALUATION_PATH = "outputs/evaluation.json"

EXPERIMENT_NAME = "predictive-maintenance"
MODEL_NAME = "predictive-maintenance"
ALIAS = "champion"

TARGET = "Machine failure"
DROP_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
TRACKED_INPUTS = ["src", "params.yaml", "dvc.yaml", "data/raw/data.csv.dvc"]


def load_params(path):
    with open(path) as f: 
        return yaml.safe_load(f)


def split_xy(df):
    X = df.drop(columns = TARGET)
    y = df[TARGET]

    return X, y


def require_tracking_uri():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("URI MLflow n'est pas definit")


def git_is_clean(paths):
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() == ""


def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()