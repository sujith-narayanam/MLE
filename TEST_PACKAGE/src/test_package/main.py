import mlflow
from ingest_data import start_ingest_data
from score import start_score
from train import start_train

mlflow.end_run()
with mlflow.start_run(run_name="Housing_Price") as parent_run:
    with mlflow.start_run(run_name="ingest_data", nested=True) as child_run_ingest:
        start_ingest_data()
    with mlflow.start_run(run_name="train", nested=True) as child_run_train:
        start_train()
    with mlflow.start_run(run_name="score", nested=True) as child_run_score:
        start_score()
