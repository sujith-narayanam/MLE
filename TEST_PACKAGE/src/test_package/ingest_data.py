"""
A script (ingest_data.py)to download and create training and validation datasets.
The script should accept the output folder/file path as an user argument.

"""


def start_ingest_data():

    import argparse
    import logging
    import os
    import tarfile

    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from six.moves import urllib
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    logging.basicConfig(
        filename=os.path.join("..", "..", "logs", "ingest_data_logs.log"),
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_path",
        default="processed",
        help="specifies the output folder",
        type=str,
    )
    parser.add_argument(
        "model_path",
        default="models",
        help="specifies the model folder",
        type=str,
    )
    args = parser.parse_args()
    OUT_PATH = os.path.join("..", "..", "data", args.out_path)
    MOD_PATH = os.path.join("..", "..", "artifacts", args.model_path)

    logging.info(f"out path: {OUT_PATH}")
    logging.info(f"mod path: {MOD_PATH}")
    logging.warning("started executing ingest_data module")

    # DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/tree/master/"
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join("..", "..", "data", "raw")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        logging.debug("extracting data from the tarfile")
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        logging.debug("extraction completed")

    def load_housing_data(housing_path=HOUSING_PATH):
        logging.debug("started loading data")
        csv_path = os.path.join(housing_path, "housing.csv")
        logging.debug("completed loading data")
        return pd.read_csv(csv_path)

    def save_housing_data(df, filename, out_path=OUT_PATH):
        logging.debug("started saving data file")
        csv_path = os.path.join(out_path, filename)
        df.to_csv(csv_path)
        logging.debug("completed saving data file")

    logging.info("starting fetch_housing_data method")
    fetch_housing_data()
    housing = load_housing_data()
    logging.info("complete fetch_housing_data method")

    logging.debug("splitting the data")

    test_size = 0.20
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("train_size", 1 - test_size)

    train_set, test_set = train_test_split(
        housing, test_size=test_size, random_state=42
    )

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    hous_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    hous_tr["rooms_per_household"] = hous_tr["total_rooms"] / hous_tr["households"]
    hous_tr["bedrooms_per_room"] = hous_tr["total_bedrooms"] / hous_tr["total_rooms"]
    hous_tr["population_per_household"] = hous_tr["population"] / hous_tr["households"]

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = hous_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    housing_prepared = housing_prepared.join(housing_labels)

    housing = strat_test_set.drop("median_house_value", axis=1)
    housing_labels = strat_test_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    hous_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    hous_tr["rooms_per_household"] = hous_tr["total_rooms"] / hous_tr["households"]
    hous_tr["bedrooms_per_room"] = hous_tr["total_bedrooms"] / hous_tr["total_rooms"]
    hous_tr["population_per_household"] = hous_tr["population"] / hous_tr["households"]

    housing_cat = housing[["ocean_proximity"]]
    housing_test = hous_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    housing_test = housing_test.join(housing_labels)

    logging.debug("started saving the processed data")
    save_housing_data(df=housing_prepared, filename="train.csv")
    save_housing_data(df=housing_test, filename="test.csv")
    logging.debug("completed saving the processed data")

    logging.debug(f"final shape of housing_prepared: {housing_prepared.shape}")
    logging.debug(f"final shape of housing_test: {housing_test.shape}")
    logging.warning("completed executing ingest_data module")

    logging.debug("tracking raw, train and test data using mlflow")
    mlflow.log_artifact(local_path=os.path.join("..", "..", "data"))
    logging.debug("finished tracking raw, train and test data in artifacts of mlflow")
