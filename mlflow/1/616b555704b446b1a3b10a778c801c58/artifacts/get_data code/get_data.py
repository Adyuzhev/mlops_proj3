from sklearn import datasets
import pandas as pd
import mlflow
import os


os.environ["MLFLOW_REGISTRY_URI"] = "/home/lesha/mlops_proj3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

with mlflow.start_run():
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']

    df = pd.DataFrame(x)
    df['target'] = y
    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/scripts/get_data.py",
                        artifact_path="get_data code")

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/datasets/raw_data.csv",
                        artifact_path="raw data")

    mlflow.end_run()
df.to_csv('~/mlops_proj3/datasets/raw_data.csv', index=False)

