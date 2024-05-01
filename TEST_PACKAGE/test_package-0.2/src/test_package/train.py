# importing libraries
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import six
import sklearn
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor


def income_cat_proportions(data):
    """gives proportion of each unique value present in income cat

    Args:
        data (dataframe): dataframes for which the proportion of income_category column is to be calculated

    Returns:
        series: proportions of income_categories are provided as output
    """
    return data["income_cat"].value_counts() / len(data)


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "input_path",
#     type=str,
#     help="output path of the ingest script will be stored at this location",
# )


# parser.add_argument(
#     "output_path",
#     type=str,
#     help="output path of the train script will be stored at this location",
# )

# args = parser.parse_args()
# output_path = args.output_path
# input_path = args.input_path

# print("input_path")
# input_path = str("C:/Users/sujith.narayana/Downloads/MLE-main/MLE-main/TEST_PACKAGE/data/ingest")
# print("output_path")
# output_path = str("C:/Users/sujith.narayana/Downloads/MLE-main/MLE-main/TEST_PACKAGE/data/train")


def train(input_path, output_path):
    train_set = pd.read_csv(input_path + "/train_set.csv")
    test_set = pd.read_csv(input_path + "/test_set.csv")
    strat_train_set = pd.read_csv(input_path + "/strat_train_set.csv")
    strat_test_set = pd.read_csv(input_path + "/strat_test_set.csv")
    housing = pd.read_csv(input_path + "/housing_1.csv")

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()

    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    housing["dummy"] = housing["total_rooms"] / housing["households"]
    housing.rename(columns={"dummy": "rooms_per_household"}, inplace=True)

    housing["dummy"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing.rename(columns={"dummy": "bedrooms_per_room"}, inplace=True)

    housing["dummy"] = housing["population"] / housing["households"]
    housing.rename(columns={"dummy": "population_per_household"}, inplace=True)

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop(
        ["ocean_proximity", "Unnamed: 0"], axis=1, errors="ignore"
    )

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

    housing_tr["dummy"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr.rename(columns={"dummy": "rooms_per_household"}, inplace=True)
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    Dummy_df = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(Dummy_df, drop_first=True))
    housing_cat = Dummy_df.copy()

    lin_reg = sklearn.linear_model.LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = sklearn.model_selection.RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = sklearn.model_selection.GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    pickle.dump(lin_reg, open(output_path + "/lin_reg.pkl", "wb"))
    pickle.dump(tree_reg, open(output_path + "/tree_reg.pkl", "wb"))
    pickle.dump(final_model, open(output_path + "/final_model.pkl", "wb"))
    pickle.dump(rnd_search, open(output_path + "/rnd_search.pkl", "wb"))
    pickle.dump(imputer, open(output_path + "/imputer.pkl", "wb"))
    pickle.dump(grid_search, open(output_path + "/grid_search.pkl", "wb"))

    housing_labels.to_csv(output_path + "/housing_labels.csv", index=False)
    housing_prepared.to_csv(output_path + "/housing_prepared.csv", index=False)
    strat_test_set.to_csv(output_path + "/strat_test_set.csv", index=False)
