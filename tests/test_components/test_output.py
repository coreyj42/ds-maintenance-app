import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from ds_maintenance_app.components.output import (
    predict_oil_change_date_from_forecast,
    calculate_latest_service_distances,
    estimate_future_service_distance,
    get_upcoming_service_dates,
    get_last_serviceable_date,
)


# Forecast results with deltas for postprocessing
@pytest.fixture
def sample_forecast_results():
    df = pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2", "id2", "id3"],
            "ds": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                ]
            ),
            "low": [10, 5, -1, -1, -2, 4],
        }
    )
    return df


def test_predict_oil_change_date_from_forecast(sample_forecast_results):
    df_result = predict_oil_change_date_from_forecast(sample_forecast_results)

    # Expected: only the first date per unique_id where 'low' <= 0
    expected = (
        pd.DataFrame(
            {
                "unique_id": ["id2", "id1"],
                "ds": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            }
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )

    # Reset index to ignore index differences
    df_result = df_result.reset_index(drop=True)

    # Assert that the function output matches expected
    assert_frame_equal(df_result, expected)


@pytest.fixture
def sample_df_features_for_lost_distance():
    df = pd.DataFrame(
        {
            "vin": ["id1", "id1", "id2", "id2"],
            "updated_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-05",  # latest for id1
                    "2024-01-02",
                    "2024-01-05",  # latest for id2
                ]
            ),
            "service_distance": [1000, 1000, 500, 500],
            "gmx_odometer": [10000, 10200, 20000, 20100],
        }
    )
    return df


def test_calculate_latest_service_distances(sample_df_features_for_lost_distance):
    df_result = calculate_latest_service_distances(
        df_features=sample_df_features_for_lost_distance,
        service_distance_column="service_distance",
    )

    # Expected output from running calculate_latest_service_distances()
    expected = (
        pd.DataFrame(
            {
                "unique_id": ["id1", "id2"],
                "service_distance_last_recorded": [1000, 500],
            }
        )
        .sort_values("unique_id")
        .reset_index(drop=True)
    )

    # Assert result is equal to expected output
    assert_frame_equal(df_result, expected)


@pytest.fixture
def sample_estimate_future_inputs():
    # Forecasted odometer deltas
    df_forecast = pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2", "id2"],
            "ds": pd.to_datetime(
                [
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-06",
                    "2024-01-07",
                ]
            ),
            "high_delta": [50, 60, 40, 10, 20],
        }
    )

    # Latest recorded service distance per vehicle
    df_service_latest = pd.DataFrame(
        {
            "unique_id": ["id1", "id2"],
            "service_distance_last_recorded": [1000, 500],
        }
    )

    return df_forecast, df_service_latest


def test_estimate_future_service_distance(sample_estimate_future_inputs):
    df_forecast, df_service_latest = sample_estimate_future_inputs

    df_result = estimate_future_service_distance(
        df_forecast_results_odometer_post=df_forecast,
        df_service_distance_latest=df_service_latest,
        service_distance_column="service_distance",
        target_delta="high_delta",
    )

    # Build expected output manually
    expected = df_forecast.merge(df_service_latest, on="unique_id", how="left")

    # Add expected forecast for decaying service distance values
    expected["service_distance_estimate"] = [950, 890, 850, 490, 470]

    # Sort to ensure deterministic comparison
    df_result = df_result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    expected = expected.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Assert result matches expected output
    assert_frame_equal(df_result, expected)


@pytest.fixture
def sample_service_forecast():
    df = pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id2", "id2", "id3"],
            "ds": pd.to_datetime(
                [
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-06",
                ]
            ),
            "service_distance_forecast": [10, -5, -1, -3, 5],
        }
    )
    return df


def test_get_upcoming_service_dates(sample_service_forecast):
    df_result = get_upcoming_service_dates(
        df=sample_service_forecast,
        service_distance_forecast_column="service_distance_forecast",
    )

    # Manually build expected output:
    # id1: first ds where forecast <= 0 -> 2024-01-07
    # id2: first ds where forecast <= 0 -> 2024-01-05
    # id3: never <= 0 -> excluded
    expected = (
        pd.DataFrame(
            {
                "unique_id": ["id2", "id1"],
                "ds": pd.to_datetime(["2024-01-05", "2024-01-07"]),
            }
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )

    df_result = df_result.reset_index(drop=True)

    # Assert results match expected output
    assert_frame_equal(df_result, expected)


@pytest.fixture
def sample_future_and_services():
    # Future forecast features
    df_future = pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2", "id2", "id3"],
            "ds": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                ]
            ),
            "is_booked": [0, 0, 0, 0, 1, 0],
        }
    )

    # Upcoming service predictions
    df_services = pd.DataFrame(
        {
            "unique_id": ["id1", "id2", "id3"],
            "ds_service_needed": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-05"]
            ),
        }
    )

    return df_future, df_services


def test_get_last_serviceable_date(sample_future_and_services):
    df_future, df_services = sample_future_and_services

    df_result = get_last_serviceable_date(df_services, df_future)

    # Expected last serviceable dates:
    # id1: ds < 2024-01-03 and is_booked == 0 -> max(ds) = 2024-01-02
    # id2: ds < 2024-01-02 and is_booked == 0 -> max(ds) = 2024-01-01
    # id3: ds < 2024-01-05 and is_booked == 0 -> max(ds) = 2024-01-01
    expected = pd.DataFrame(
        {
            "unique_id": ["id1", "id2", "id3"],
            "ds_service_needed": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-05"]
            ),
            "last_serviceable_date": pd.to_datetime(
                ["2024-01-02", "2024-01-01", "2024-01-01"]
            ),
        }
    )

    # Reset index for comparison
    df_result = df_result.sort_values("unique_id").reset_index(drop=True)
    expected = expected.sort_values("unique_id").reset_index(drop=True)

    # Assert the function output matches expected
    assert_frame_equal(df_result, expected)
