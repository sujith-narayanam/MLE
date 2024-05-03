import pandas as pd
from sklearn.model_selection import train_test_split


def test_train_test_split():
    df = pd.read_csv(
        "C:/Users/sujith.narayana/Downloads/MLE-main/MLE-main/TEST_PACKAGE/data/ingest/housing_1.csv"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("income_cat", axis=1), df["income_cat"], test_size=0.2, random_state=42
    )
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    assert len(X_train) == int(0.8 * len(df))
    assert len(X_test) == int(0.2 * len(df))
    assert set(X_train.columns) == set(df.drop("income_cat", axis=1).columns)
    assert set(X_test.columns) == set(df.drop("income_cat", axis=1).columns)
    assert "income_cat" in y_train.columns
    assert "income_cat" in y_test.columns
