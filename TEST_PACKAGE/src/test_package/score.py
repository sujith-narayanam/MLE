"""
A script (score.py) to score the model(s).
The script should accept arguments for model folder, dataset folder and any outputs.

"""


def start_score():
    import argparse
    import logging
    import os
    import pickle

    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)

    logging.basicConfig(
        filename=os.path.join("..", "..", "logs", "score_logs.log"),
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        default="processed",
        help="specifies the data folder",
        type=str,
    )
    parser.add_argument(
        "model_path",
        default="models",
        help="specifies the model folder",
        type=str,
    )
    args = parser.parse_args()
    MOD_PATH = os.path.join("..", "..", "artifacts", args.model_path)
    DATA_PATH = os.path.join("..", "..", "data", args.data_path)

    logging.warning("started executing score module")

    def load_housing_data(housing_path=DATA_PATH):
        csv_path = os.path.join(housing_path, "test.csv")
        return pd.read_csv(csv_path)

    logging.debug("loading house data into data frame")
    housing = load_housing_data()
    logging.debug("completed loading data in to data frame")

    housing_labels = housing["median_house_value"]
    del housing["median_house_value"]
    housing_prepared = housing.copy()

    def score(y_test, y_pred):
        logging.debug("inside score function")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logging.debug("exiting score function")
        return {mse, rmse, r2, mae}

    # check all models
    for model in os.listdir(MOD_PATH):
        filename = model
        loaded_model = pickle.load(
            open(os.path.join(MOD_PATH, filename), "rb")
        )
        y_pred = loaded_model.predict(housing_prepared)
        logging.info(
            "calculating score for {}".format(
                " ".join(model.split(".")[0].upper().split("_"))
            )
        )
        mse, rmse, r2, mae = score(housing_labels, y_pred)
        # print()
        # print(" ".join(model.split(".")[0].upper().split("_")) + " results: ")
        temp = " ".join(model.split(".")[0].upper().split("_"))
        logging.info(f"Mean Squared Error for {temp} is {mse}")
        logging.info(f"Root Mean Squared Error for {temp} is {rmse}")
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        # print("MSE : ", mse)
        # print("RMSE : ", rmse)

    logging.warning("completed executing score module")
