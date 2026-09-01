import pytest
import pandas as pd
import numpy as np

from ds_maintenance_app.components.forecast import (
    prepare_features_for_forecaster,
    prepare_future_table,
    first_order_difference_target,
    build_forecasters,
    execute_forecasters,
    backtest,
)


# DataFrame of future data to split
@pytest.fixture
def sample_df_prepare():

    # Build date values for df
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")

    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 100,
            "x": [np.nan] + [1] * 99,
            "date": list(dates),
            "dynamic_feature": [1] * 100,
            "static_feature": [1] * 100,
            "unused_feature": [1] * 100,
        }
    )
    return df_test


def test_prepare_features_for_forecaster(sample_df_prepare):
    df_result = prepare_features_for_forecaster(
        sample_df_prepare, "x", "vin", "date", ["static_feature"], ["dynamic_feature"]
    )

    expected_columns = {
        "unique_id",
        "y",
        "ds",
        "static_feature",
        "dynamic_feature",
    }

    # Assert column names are prepared correctly
    assert set(df_result.columns) == expected_columns

    # Assert result is sorted
    assert df_result.equals(
        df_result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    )

    # Assert nulls have been dropped
    assert len(df_result[df_result["y"].isna()]) == 0


def test_prepare_future_table(sample_df_prepare):
    df_result = prepare_future_table(
        sample_df_prepare, "vin", "date", ["dynamic_feature"]
    )

    expected_columns = {
        "unique_id",
        "ds",
        "dynamic_feature",
    }

    # Assert column names are prepared correctly
    assert set(df_result.columns) == expected_columns


@pytest.fixture
def sample_df_stats():

    # Build date values of different length that end on the same day
    dates1 = pd.date_range(end="2025-01-01", periods=100, freq="D")
    dates2 = pd.date_range(end="2025-01-01", periods=50, freq="D")
    dates = list(dates1) + list(dates2)

    df_test = pd.DataFrame(
        {
            "unique_id": ["id1"] * 100 + ["id2"] * 50,
            "y": list(range(50)) + list(range(149, 199)) + list(range(50)),
            "ds": dates,
            "dynamic_feature": list(range(150)),
            "static_feature": [1] * 150,
        }
    )
    return df_test


@pytest.fixture
def sample_df_future(sample_df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Future dataset starting from the last date of sample_df_stats
    and extending 10 days forward (daily frequency).
    """
    last_date = sample_df_stats["ds"].max()

    # Start from next day
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=10, freq="D"
    )

    df_future = pd.DataFrame(
        {
            "unique_id": ["id1"] * 10 + ["id2"] * 10,
            "ds": list(future_dates) * 2,
            "dynamic_feature": range(100, 120),
        }
    )

    return df_future


def test_first_order_difference_target(sample_df_stats):

    df_result = first_order_difference_target(sample_df_stats)

    # Assert differencing applied to y
    assert df_result["y"].max() == 100
    assert df_result["y"].iloc[0] == 0
    assert df_result["y"].iloc[-50] == 0
    for x in df_result[(df_result["y"] != 100) & (df_result["y"] != 0)]["y"].values:
        assert x == 1


def test_build_forecasters(sample_df_stats):

    # Build a test forecaster
    forecasters = build_forecasters(
        sample_df_stats,
        10,
        [6, 0],
        {"enable_categorical": True},
        [1],
        date_features=["week"],
        static_features=["static_feature"],
    )

    # Assert forecaster dict output is correct
    assert len(forecasters) == 2
    assert set(forecasters.keys()) == {6, 0}


def test_execute_forecasters(sample_df_stats, sample_df_future):

    # Number of windows to use for CI calculations in each forecaster
    n_windows_list = [6, 0]

    # Horizon to forecast
    horizon = 10

    # Build a test forecaster
    forecasters = build_forecasters(
        sample_df_stats,
        10,
        n_windows_list,
        {"enable_categorical": True},
        [1],
        date_features=["week"],
        static_features=["static_feature"],
    )

    forecast_results = execute_forecasters(
        forecasters, n_windows_list, sample_df_future, horizon, level=20
    )

    # Expected columns in MLModels backtest output
    expected_cols = {"unique_id", "ds", "yhat"}
    assert expected_cols.issubset(set(forecast_results.columns))

    # Assert all unique_ids present
    assert set(forecast_results["unique_id"].unique()) == {"id1", "id2"}

    # Assert predictions are numeric
    assert pd.api.types.is_numeric_dtype(forecast_results["yhat"])


def test_backtest(sample_df_stats):

    # Run backtest
    df_backtest = backtest(
        df_stats_delta=sample_df_stats,
        horizon=5,
        n_windows=1,
        model_hyperparameters={"enable_categorical": True},
        lags=[1],
        date_features=["week"],
        static_features=["static_feature"],
    )

    # Expected columns in MLModels backtest output
    expected_cols = {"unique_id", "ds", "yhat"}
    assert expected_cols.issubset(set(df_backtest.columns))

    # Assert all unique_ids present
    assert set(df_backtest["unique_id"].unique()) == {"id1", "id2"}

    # Assert predictions are numeric
    assert pd.api.types.is_numeric_dtype(df_backtest["yhat"])
