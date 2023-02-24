# importing libraries
import argparse
import os
import tarfile

import numpy as np
import pandas as pd
import six
import sklearn
from sklearn.model_selection import train_test_split


def fetch_housing_data(housing_url, housing_path):
    """extracting housing data

    Args:
        housing_url (str): link from which data is being read
        housing_path (str): offline path for reading data
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = housing_path + "/housing.tgz"  # os.path.join
    six.moves.urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """reading housing data

    Args:
        housing_path (str): offline path for reading data

    Returns:
        dataframe: dataset that is being read
    """
    csv_path = housing_path + "/housing.csv"
    return pd.read_csv(csv_path)


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "output_path",
#     type=str,
#     help="output path of the ingest_data script will be stored at this location",
# )

# args = parser.parse_args()
# output_path = args.output_path


def ingest_data(output_path):
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "datasets/housing"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    fetch_housing_data(HOUSING_URL, HOUSING_PATH)

    housing = load_housing_data(HOUSING_PATH)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = sklearn.model_selection.train_test_split(
        housing, test_size=0.2, random_state=42
    )

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    train_set.to_csv(output_path + "/train_set.csv", index=False)
    test_set.to_csv(output_path + "/test_set.csv", index=False)
    strat_train_set.to_csv(output_path + "/strat_train_set.csv", index=False)
    strat_test_set.to_csv(output_path + "/strat_test_set.csv", index=False)
    housing.to_csv(output_path + "/housing_1.csv", index=False)
