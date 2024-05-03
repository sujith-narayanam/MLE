# importing libraries
import argparse
import os
import pickle
import tarfile

import numpy as np
import pandas as pd
import six
import sklearn
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "input_path",
#     type=str,
#     help="output path of the train_data script will be stored at this location",
# )

# args = parser.parse_args()
# input_path1 = args.input_path

def score(input_path1):
    housing_labels = pd.read_csv(input_path1 + "/housing_labels.csv")
    housing_prepared = pd.read_csv(input_path1 + "/housing_prepared.csv")
    strat_test_set = pd.read_csv(input_path1 + "/strat_test_set.csv")

    lin_reg = pickle.load(open(input_path1 + "/lin_reg.pkl", "rb"))
    tree_reg = pickle.load(open(input_path1 + "/tree_reg.pkl", "rb"))
    final_model = pickle.load(open(input_path1 + "/final_model.pkl", "rb"))
    rnd_search = pickle.load(open(input_path1 + "/rnd_search.pkl", "rb"))
    imputer = pickle.load(open(input_path1 + "/imputer.pkl", "rb"))
    grid_search = pickle.load(open(input_path1 + "/grid_search.pkl", "rb"))

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    lin_mae

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse

    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


    X_test = strat_test_set.drop(
        ["median_house_value", "Unnamed: 0"], axis=1, errors="ignore"
    )
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )


    Dup_df = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(Dup_df, drop_first=True))
    X_test_cat = Dup_df.copy()


    final_predictions = final_model.predict(X_test_prepared)
    final_mse = sklearn.metrics.mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
