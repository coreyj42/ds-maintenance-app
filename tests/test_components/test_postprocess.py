import pytest
import pandas as pd
import numpy as np

from ds_maintenance_app.utils.utils import load_config
from ds_maintenance_app.components.postprocess import (
    _merge_future_features,
    _fill_missing_intervals,
    _apply_conditional_flattening,
    _clip_to_trend,
    _merge_last_observation,
    postprocess_forecast,
    postprocess_backtest,
)


# Forecast results with deltas for postprocessing
@pytest.fixture
def sample_forecast_results():
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": ["id1"] * 5,
            "ds": dates,
            "yhat": [0, 2, 3, 4, 5],
            "low": [-0.7, np.nan, 2.8, 3.8, np.nan],
            "high": [0.9, np.nan, 3.2, 4.2, np.nan],
        }
    )
    return df


# Future features to merge
@pytest.fixture
def sample_future_features():
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {"unique_id": ["id1"] * 5, "ds": dates, "dynamic_feature": [1, 1, 1, 0, 0]}
    )
    return df


# Historical stats with last observed values
@pytest.fixture
def sample_historical_stats():
    dates = pd.date_range(start="2024-12-27", periods=5, freq="D")
    df = pd.DataFrame(
        {"unique_id": ["id1"] * 5, "y": [100, 101, 102, 103, 104], "ds": dates}
    )
    return df


# Postprocessing dictionary for conditional flattening
@pytest.fixture
def sample_postprocess_dict():
    return load_config("config_tests.yaml")["postprocess"]


def test_merge_future_features(sample_forecast_results, sample_future_features):
    dynamic_features = ["dynamic_feature"]

    df_merged = _merge_future_features(
        df_forecast=sample_forecast_results,
        X_future=sample_future_features,
        dynamic_features=dynamic_features,
    )

    # Assert that the merged DataFrame contains the dynamic feature column
    assert all(
        col in df_merged.columns for col in ["yhat", "low", "high", "dynamic_feature"]
    )

    # Assert that dynamic features are correctly merged by checking some values
    expected_values = sample_future_features["dynamic_feature"].values
    actual_values = df_merged["dynamic_feature"].values
    assert all(expected_values == actual_values)

    # Assert the shape remains consistent with forecast rows
    assert df_merged.shape[0] == sample_forecast_results.shape[0]


def test_fill_missing_intervals(sample_forecast_results):
    df_filled = _fill_missing_intervals(sample_forecast_results)

    # Assert no missing values remain in 'low' and 'high'
    assert df_filled["low"].isna().sum() == 0
    assert df_filled["high"].isna().sum() == 0

    # Assert that rows which were NaN are now equal to yhat
    mask = sample_forecast_results["low"].isna()
    assert all(df_filled.loc[mask, "low"] == sample_forecast_results.loc[mask, "yhat"])
    assert all(df_filled.loc[mask, "high"] == sample_forecast_results.loc[mask, "yhat"])

    # Assert that rows which were not NaN remain unchanged
    mask_notna = ~sample_forecast_results["low"].isna()
    assert all(
        df_filled.loc[mask_notna, "low"]
        == sample_forecast_results.loc[mask_notna, "low"]
    )
    assert all(
        df_filled.loc[mask_notna, "high"]
        == sample_forecast_results.loc[mask_notna, "high"]
    )


# Apply postprocessing before conditional flattening
def _apply_postprocessing_before_conditional_flattening(
    sample_forecast_results, sample_future_features
):
    # First, merge a dynamic feature to test conditional flattening
    df_with_future = sample_forecast_results.merge(
        sample_future_features, on=["unique_id", "ds"]
    )

    # Fill missing prediction confidence interval values
    df_with_future = _fill_missing_intervals(df_with_future)

    return df_with_future


def test_apply_conditional_flattening(
    sample_forecast_results, sample_postprocess_dict, sample_future_features
):

    # Apply postprocessing until stage needed for test
    df_with_future = _apply_postprocessing_before_conditional_flattening(
        sample_forecast_results, sample_future_features
    )

    df_flattened = _apply_conditional_flattening(
        df=df_with_future, postprocess_dict=sample_postprocess_dict
    )

    feature = list(sample_postprocess_dict.keys())[0]
    value = sample_postprocess_dict[feature]

    # Rows matching the condition should have yhat, low, high set to 0
    mask = df_with_future[feature] == value
    assert all(df_flattened.loc[mask, ["yhat", "low", "high"]].values.flatten() == 0.0)

    # Rows not matching the condition should remain unchanged
    mask_not = df_with_future[feature] != value
    assert all(
        df_flattened.loc[mask_not, ["yhat", "low", "high"]].values.flatten()
        == df_with_future.loc[mask_not, ["yhat", "low", "high"]].values.flatten()
    )


# Apply postprocessing before clip_to_trend
def _apply_postprocessing_before_clip_to_trend(
    sample_forecast_results, sample_postprocess_dict, sample_future_features
):

    # Apply postprocessing up to stage before conditional flattening
    df_with_future = _apply_postprocessing_before_conditional_flattening(
        sample_forecast_results, sample_future_features
    )

    df_flattened = _apply_conditional_flattening(
        df=df_with_future, postprocess_dict=sample_postprocess_dict
    )

    return df_flattened


def test_clip_to_trend(
    sample_forecast_results, sample_postprocess_dict, sample_future_features
):

    # Apply postprocessing up to stage needed for test
    df_flattened = _apply_postprocessing_before_clip_to_trend(
        sample_forecast_results, sample_postprocess_dict, sample_future_features
    )

    # Test trend="up" → all values should be >= 0
    df_up = _clip_to_trend(df_flattened, trend="up")
    assert (df_up[["yhat", "low", "high"]] >= 0).all().all()

    # Test trend="down" → all values should be <= 0
    df_down = _clip_to_trend(df_flattened, trend="down")
    assert (df_down[["yhat", "low", "high"]] <= 0).all().all()

    # Test trend=None → values remain unchanged
    df_none = _clip_to_trend(sample_forecast_results, trend=None)
    pd.testing.assert_frame_equal(df_none, sample_forecast_results)


# Apply postprocessing before merge_last_observation
def _apply_postprocessing_before_merge_last_observation(
    sample_forecast_results, sample_postprocess_dict, sample_future_features
):
    # Apply postprocessing up to stage before trend clipping
    df_flattened = _apply_postprocessing_before_clip_to_trend(
        sample_forecast_results, sample_postprocess_dict, sample_future_features
    )

    # Apply trend clipping to upward trend
    df_up = _clip_to_trend(df_flattened, trend="up")

    return df_up


def test_merge_last_observation(
    sample_forecast_results,
    sample_postprocess_dict,
    sample_future_features,
    sample_historical_stats,
):

    # Apply postprocessing steps needed before _merge_last_observation
    df_up = _apply_postprocessing_before_merge_last_observation(
        sample_forecast_results, sample_postprocess_dict, sample_future_features
    )

    # Merge last observed value with predictions
    df_merged = _merge_last_observation(df=df_up, df_stats=sample_historical_stats)

    # Assert the new column y_last exists
    assert "y_last" in df_merged.columns

    # Assert all rows have the last observed value from historical stats
    last_value = sample_historical_stats["y"].iloc[-1]

    assert all(df_merged["y_last"] == last_value)


# Test full postprocess pipeline function
def test_postprocess_forecast(
    sample_forecast_results,
    sample_postprocess_dict,
    sample_future_features,
    sample_historical_stats,
):

    dynamic_features = ["dynamic_feature"]
    trend = "up"

    df_post = postprocess_forecast(
        sample_forecast_results,
        sample_future_features,
        sample_historical_stats,
        dynamic_features,
        sample_postprocess_dict,
        trend,
    )

    # Check that merged dynamic features exist
    for feature in dynamic_features:
        assert feature in df_post.columns

    # Check that y_last is correctly merged
    last_value = sample_historical_stats["y"].iloc[-1]
    assert all(df_post["y_last"] == last_value)

    # Check that delta columns exist
    for col in ["yhat", "low", "high"]:
        delta_col = f"{col}_delta"

        # Assert that delta columns are present in final df
        assert delta_col in df_post.columns

    # Check that cumulative reconstruction is correct
    # yhat should equal yhat_delta cumsum + y_last
    expected_yhat = df_post["yhat_delta"].cumsum() + df_post["y_last"].iloc[0]
    assert all(df_post["yhat"] == expected_yhat)


# Mock backtest results
@pytest.fixture
def df_backtest_results_sample():
    df = pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id2", "id2"],
            "ds": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-02", "2025-01-03"]
            ),
            "cutoff": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01"]
            ),
            "yhat": [1, 1, 2, 2],
        }
    )
    return df


# Mock df_stats
@pytest.fixture
def df_stats_sample():
    df = pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2", "id2", "id2"],
            "ds": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "y": [100, 101, 102, 200, 201, 202],
            "dynamic_feature": [1, 1, 1, 0, 0, 1],
        }
    )
    return df


def test_postprocess_backtest(
    df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
):

    df_backtest_results_post = postprocess_backtest(
        df_backtest_results=df_backtest_results_sample,
        df_stats=df_stats_sample,
        dynamic_features=["dynamic_feature"],
        postprocess_dict=sample_postprocess_dict,
    )

    # Assert original columns still exist
    for col in ["unique_id", "ds", "cutoff", "yhat", "y_last", "yhat_delta"]:
        assert col in df_backtest_results_post.columns

    # Check y_cutoff values are correctly merged
    expected_y_cutoff = [100, 100, 200, 200]
    assert all(df_backtest_results_post["y_last"].values == expected_y_cutoff)

    # Check cumulative reconstruction: yhat should be delta + y_cutoff (since yhat values are all positive)
    assert all(df_backtest_results_post["yhat"] >= df_backtest_results_post["y_last"])

    # Check that delta column exists
    assert "yhat_delta" in df_backtest_results_post.columns
