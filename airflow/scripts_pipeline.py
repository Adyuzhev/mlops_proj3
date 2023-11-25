from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    "owner": "admin",
    "start_date": dt.datetime(2023, 11, 25),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=2),
    "depends_on_past": False
}

with DAG(
    dag_id='scripts_pipeline',
    default_args=args,
    schedule_interval=None,
    tags=['score'],
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="python3 /home/lesha/mlops_proj3/scripts/get_data.py", 
                            dag=dag)
    train_test_split = BashOperator(task_id='train_test_split',
                            bash_command="python3 /home/lesha/mlops_proj3/scripts/train_test_split.py", 
                            dag=dag)
    normalization = BashOperator(task_id='normalization',
                            bash_command="python3 /home/lesha/mlops_proj3/scripts/normalization.py", 
                            dag=dag)  
    data_train = BashOperator(task_id='data_train',
                            bash_command="python3 /home/lesha/mlops_proj3/scripts/data_train.py", 
                            dag=dag)
    evaluate = BashOperator(task_id='evaluate',
                            bash_command="python3 /home/lesha/mlops_proj3/scripts/evaluate.py", 
                            dag=dag)
    get_data >> train_test_split >> normalization >> data_train >> evaluate
