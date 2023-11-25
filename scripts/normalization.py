import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import mlflow


os.environ["MLFLOW_REGISTRY_URI"] = "/home/lesha/mlops_proj3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("normalization")

with mlflow.start_run():
    df_train = pd.read_csv('~/mlops_proj3/datasets/train.csv')
    X_train = df_train.drop(columns='target')
    y_train = df_train['target'].reset_index(drop=True)

    df_test = pd.read_csv('~/mlops_proj3/datasets/test.csv')
    X_test = df_test.drop(columns='target')
    y_test = df_test['target'].reset_index(drop=True)

    scaler = "MinMaxScaler"

    if scaler == "MinMaxScaler":
        mm_scaler = MinMaxScaler()
        X_train_scaled = mm_scaler.fit_transform(X_train)
        X_test_scaled = mm_scaler.transform(X_test)

    elif scaler == "StandardScaler":
        st_scaler = StandardScaler()
        X_train_scaled = st_scaler.fit_transform(X_train)
        X_test_scaled = st_scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled)
    X_test_scaled = pd.DataFrame(X_test_scaled)

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/scripts/normalization.py",
                        artifact_path="normalization code")

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/datasets/train_norm.csv",
                        artifact_path="train_norm data")

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/datasets/test_norm.csv",
                        artifact_path="test_norm data")

    mlflow.end_run()

pd.concat([X_train_scaled, y_train], axis=1).to_csv('~/mlops_proj3/datasets/train_norm.csv', index=None)
pd.concat([X_test_scaled, y_test], axis=1).to_csv('~/mlops_proj3/datasets/test_norm.csv', index=None)