
import pickle
import pandas as pd
import os
import mlflow


os.environ["MLFLOW_REGISTRY_URI"] = "/home/lesha/mlops_proj3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("evaluate")

with mlflow.start_run():
    df_test = pd.read_csv('~/mlops_proj3/datasets/test_norm.csv')
    X_test = df_test.drop(columns='target')
    y_test = df_test['target']

    model = pickle.load(open('/home/lesha/mlops_proj3/models/model.pkl', 'rb'))
    score = model.score(X_test, y_test)

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/scripts/evaluate.py",
                        artifact_path="evaluate code")

    mlflow.sklearn.log_model(model, "model")
    
    mlflow.log_metric("test_score", score)

    mlflow.end_run()


