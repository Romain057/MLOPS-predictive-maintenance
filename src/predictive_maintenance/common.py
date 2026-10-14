import yaml

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

def load_params(path):
    with open(path) as f: 
        return yaml.safe_load(f)


def split_xy(df):
    df = df.drop(columns = DROP_COLUMNS)

    X = df.drop(columns = TARGET)
    y = df[TARGET]

    return X, y
