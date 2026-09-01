import pytest
import pandas as pd

from ds_maintenance_app.utils.utils import load_config
from ds_maintenance_app.components.postprocess import postprocess_backtest
from ds_maintenance_app.components.evaluate import (
    calculate_mae,
    calculate_mean_absolute_scaled_error_h,
    calculate_wmape,
    calculate_average_bias,
    calculate_global_bias,
    calculate_evaluation_metrics,
)


# Postprocessing dictionary for conditional flattening
@pytest.fixture
def sample_postprocess_dict():
    return load_config("config_tests.yaml")["postprocess"]


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


# Postprocess backtest result
def _postprocess_backtest(
    df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
):

    df_backtest_results_post = postprocess_backtest(
        df_backtest_results=df_backtest_results_sample,
        df_stats=df_stats_sample,
        dynamic_features=["dynamic_feature"],
        postprocess_dict=sample_postprocess_dict,
    )

    return df_backtest_results_post


def test_calculate_mae(
    df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
):

    # Apply postprocessing to backtest result df
    df_backtest_results_post = _postprocess_backtest(
        df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
    )

    # Calculate MASE of backtest results
    mase_result = calculate_mae(df_backtest_results_post)

    assert type(mase_result) is float


def test_calculate_mean_absolute_scaled_error_h(
    df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
):

    # Apply postprocessing to backtest result df
    df_backtest_results_post = _postprocess_backtest(
        df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
    )

    # Calculate MASE of backtest results
    mase_result = calculate_mean_absolute_scaled_error_h(
        df_backtest_results_post, df_stats_sample
    )

    assert type(mase_result) is float


def test_calculate_wmape(
    df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
):

    # Apply postprocessing to backtest result df
    df_backtest_results_post = _postprocess_backtest(
        df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
    )

    # Calculate wMAPE of backtest results
    wmape_result = calculate_wmape(df_backtest_results_post)

    assert type(wmape_result) is float


def test_calculate_average_bias(
    df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
):

    # Apply postprocessing to backtest result df
    df_backtest_results_post = _postprocess_backtest(
        df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
    )

    # Calculate average bias of backtest results
    bias_result = calculate_average_bias(df_backtest_results_post)

    assert type(bias_result) is float


def test_calculate_global_bias(
    df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
):

    # Apply postprocessing to backtest result df
    df_backtest_results_post = _postprocess_backtest(
        df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
    )

    # Calculate global  bias of backtest results
    bias_result = calculate_global_bias(df_backtest_results_post)

    assert type(bias_result) is float


def test_calculate_evaluation_metrics(
    df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
):

    # Apply postprocessing to backtest result df
    df_backtest_results_post = _postprocess_backtest(
        df_backtest_results_sample, df_stats_sample, sample_postprocess_dict
    )

    # Calculate all evaluation metrics
    df_metrics = calculate_evaluation_metrics(df_backtest_results_post, df_stats_sample)

    # Assert all metrics are in results df
    expected_cols = {
        "MAE",
        "MASE",
        "wMAPE",
        "average_bias",
        "global_bias",
    }
    assert expected_cols.issubset(set(df_metrics.columns))

    # Assert only single value for each metric
    assert len(df_metrics) == 1
