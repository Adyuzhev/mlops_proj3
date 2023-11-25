import pandas as pd
from sklearn.model_selection import train_test_split
import os
import mlflow


os.environ["MLFLOW_REGISTRY_URI"] = "/home/lesha/mlops_proj3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_test_split")

with mlflow.start_run():
    df = pd.read_csv('~/mlops_proj3/datasets/raw_data.csv')
    X = df.drop(columns='target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/scripts/train_test_split.py",
                        artifact_path="train_test_split code")

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/datasets/train.csv",
                        artifact_path="train data")

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/datasets/test.csv",
                        artifact_path="test data")

    mlflow.end_run()

pd.concat([X_train, y_train], axis=1).to_csv('~/mlops_proj3/datasets/train.csv', index=None)
pd.concat([X_test, y_test], axis=1).to_csv('~/mlops_proj3/datasets/test.csv', index=None)