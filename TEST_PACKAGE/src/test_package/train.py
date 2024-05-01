"""
A script (train.py) to train the model(s).
The script should accept arguments for input (dataset) and output folders (model pickle)
"""


def start_train():
    import argparse
    import logging
    import os
    import pickle

    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from mlflow.models.signature import infer_signature
    from scipy.stats import randint
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV as rscv
    from sklearn.tree import DecisionTreeRegressor

    logging.basicConfig(
        filename=os.path.join("..", "..", "logs", "train_logs.log"),
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    logging.warning("started executing train module")

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
    DATA_PATH = os.path.join("..", "..", "data", args.data_path)
    MOD_PATH = os.path.join("..", "..", "artifacts", args.model_path)

    def load_housing_data(housing_path=DATA_PATH):
        logging.debug("started loading housing data function")
        csv_path = os.path.join(housing_path, "train.csv")
        logging.debug("completed loading housing data function")
        return pd.read_csv(csv_path)

    logging.debug("loading house data into data frame")
    housing = load_housing_data()
    logging.debug("completed loading data in to data frame")

    logging.debug(f"shape of the loaded data set: {housing.shape}")
    logging.debug(f"keys inside the loaded data set: {housing.keys()}")
    housing_labels = housing["median_house_value"]
    del housing["median_house_value"]
    housing_prepared = housing.copy()

    logging.info("Linear Regression")

    logging.debug("started training linear regression model")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    signature = infer_signature(
        housing_prepared, lin_reg.predict(housing_prepared)
    )
    mlflow.sklearn.log_model(lin_reg, "Linear_Regression", signature=signature)

    logging.debug("started saving to linear_regression pickle")
    path = os.path.join(MOD_PATH, "linear_regression.pkl")
    pickle.dump(lin_reg, open(path, "wb"))

    logging.debug("complete saving to linear_regression pickle")
    logging.debug("completed linear regression")

    logging.info("Decision Tree")

    logging.debug("start training decision tree model")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    signature = infer_signature(
        housing_prepared, tree_reg.predict(housing_prepared)
    )
    mlflow.sklearn.log_model(tree_reg, "Decision_Tree", signature=signature)

    logging.debug("started saving to decision_tree pickle")
    path = os.path.join(MOD_PATH, "decision_tree.pkl")
    pickle.dump(tree_reg, open(path, "wb"))

    logging.debug("completed saving to decision_tree pickle")
    logging.debug("completed training decision regression model")

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)

    logging.info("Random Forest With Random Search")

    logging.debug(
        "started random forest with random search with parameters: n_iter=10,cv=5"
    )
    logging.debug("using neg_mean_squared error as scoring function")
    rnd_search = rscv(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    signature = infer_signature(
        housing_prepared, rnd_search.predict(housing_prepared)
    )
    mlflow.sklearn.log_model(
        rnd_search, "Random_Forest(with random_search)", signature=signature
    )
    path = os.path.join(MOD_PATH, "random_forest_with_random_search.pkl")
    pickle.dump(rnd_search, open(path, "wb"))

    logging.debug(
        "completed saving to random forest with random search pickle"
    )
    logging.debug("completed random forest with random search")

    # cvres = rnd_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #     print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    logging.info("Random Forest With Grid Search")

    logging.debug(
        "started random forest with grid search with parameters: cv=5"
    )
    logging.debug("using neg_mean_squared_error as scoring function")
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #     print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    signature = infer_signature(
        housing_prepared, grid_search.predict(housing_prepared)
    )
    mlflow.sklearn.log_model(
        grid_search, "Random_Forest(with grid_search)", signature=signature
    )

    logging.debug("started saving to random forest with grid search pickle")
    path = os.path.join(MOD_PATH, "random_forest_with_grid_search.pkl")
    pickle.dump(grid_search, open(path, "wb"))

    logging.debug("completed saving to random forest with grid search pickle")
    logging.debug("completed random forest with grid search")

    logging.warning("complted executing train module")
