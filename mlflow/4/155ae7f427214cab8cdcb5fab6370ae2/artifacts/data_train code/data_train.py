import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os
import mlflow


os.environ["MLFLOW_REGISTRY_URI"] = "/home/lesha/mlops_proj3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("data_train")

with mlflow.start_run():

    df_train = pd.read_csv('~/mlops_proj3/datasets/train_norm.csv')
    X_train = df_train.drop(columns='target')
    y_train = df_train['target']

    logreg = LogisticRegression(penalty='l2', C=0.5, fit_intercept=True, random_state=42)
    logreg.fit(X_train, y_train)
    train_score = logreg.score(X_train, y_train)

    pickle.dump(logreg, open('/home/lesha/mlops_proj3/models/model.pkl', 'wb'))

    mlflow.log_artifact(local_path="/home/lesha/mlops_proj3/scripts/data_train.py",
                        artifact_path="data_train code")

    mlflow.sklearn.log_model(logreg, "model")
    
    mlflow.log_metric("score:", train_score)

    mlflow.end_run()


