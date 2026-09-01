import pytest
import pandas as pd
import numpy as np

from ds_maintenance_app.utils.utils import load_config
from ds_maintenance_app.components.preprocess import (
    nullify_zeros,
    nullify_if_other_not_null,
    convert_booleans,
    nullify_before_long_gaps,
    split_future_data,
    hampel_filter_forecasting_columns,
    nullify_decreases,
    nullify_all_values_before_decreases,
    nullify_before_unrealistic_changes,
    remove_resets_from_service_columns,
    interpolate_target_columns_when_booked,
    forward_fill_missing,
    drop_cars_without_measurements,
)


# DataFrame with boolean columns
@pytest.fixture
def sample_df_with_booleans():
    df_test = pd.DataFrame({"vin": ["id1", "id1"], "is_x": [True, False]})
    return df_test


# DataFrame with boolean columns
@pytest.fixture
def sample_df_with_booleans_invalid():
    df_test = pd.DataFrame({"vin": ["id1", "id1"], "is_x": ["True", "False"]})
    return df_test


def test_convert_booleans(sample_df_with_booleans):
    df_result = convert_booleans(sample_df_with_booleans)

    # Assert boolean values changed to numeric
    assert df_result["is_x"][0] == 1
    assert df_result["is_x"][1] == 0


def test_convert_booleans_invalid_input(sample_df_with_booleans_invalid):

    # Check invalid input raises error
    with pytest.raises(ValueError):
        convert_booleans(sample_df_with_booleans_invalid)


# DataFrame future data to split
@pytest.fixture
def sample_df_future():

    # Get today's date
    today = pd.Timestamp.today().normalize()  # removes time portion

    # Build date values with past and future dates
    dates = pd.date_range(
        start=today - pd.Timedelta(days=19), end=today + pd.Timedelta(days=20), freq="D"
    )

    # Build df_test
    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 40,
            "x": [1] * 40,
            "date": list(dates),
            "dynamic_feature": [1] * 40,
            "static_feature": [1] * 40,
        }
    )
    return df_test


def test_split_future_data(sample_df_future):

    df_past_result, df_future_result = split_future_data(
        sample_df_future,
        id_column="vin",
        date_column="date",
        dynamic_features=["dynamic_feature"],
    )

    # Asset sample_df_future is split into past and future
    assert len(df_past_result) == 20
    assert len(df_future_result) == 20

    # Get today's date with time 00:00
    today = pd.Timestamp.today().normalize()

    # Assert today's date is in past but not in future
    assert today in df_past_result["date"].values
    assert today not in df_future_result["date"].values

    # Asert static_features dropped from future data
    assert "static_feature" not in df_future_result.columns


# DataFrame with values to be nulled before modelling
@pytest.fixture
def sample_df_to_null():
    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 40 + ["id2"] * 40,
            "x1": list(range(40)) + list(range(0, 80, 2)),
            "x2": list(range(20, 60)) + [np.nan] * 40,
        }
    )
    return df_test


def test_nullify_zeros(sample_df_to_null):
    df_result = nullify_zeros(sample_df_to_null, measurement_columns=["x1", "x2"])

    # assert zeros have been set to None
    assert len(df_result[df_result["x1"].isna()]) == 2
    assert len(df_result[df_result["x2"].isna()]) == 40


def test_nullify_if_other_not_null(sample_df_to_null):

    # load_config("config.yaml")
    df_result = nullify_if_other_not_null(sample_df_to_null, "vin", "x1", "x2")

    # assert target values are set to None where reference values are not None
    assert len(df_result[df_result["x1"].isna()].iloc[:40]) == 40
    assert len(df_result[df_result["x1"].isna()].iloc[40:]) == 0
    assert len(df_result[df_result["x2"].isna()]) == 40


# DataFrame with large gaps in series
@pytest.fixture
def sample_df_large_gaps():

    # Build x values for valid series with small gap
    series_id1 = [1] * 40 + [np.nan] * 20 + [1] * 40

    # Build x values for series with large gap in middle
    series_id2 = [1] * 20 + [np.nan] * 60 + [1] * 20

    # Build x values for series with large gap at end
    series_id3 = [1] * 10 + [np.nan] * 90

    # Build x values for series with gap but blocking flag should stop filtering
    series_id4 = [1] * 10 + [np.nan] * 90

    # Combine series
    values = series_id1 + series_id2 + series_id3 + series_id4

    # Build date values for df
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")

    # Build df_test
    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 100 + ["id2"] * 100 + ["id3"] * 100 + ["id4"] * 100,
            "x": values,
            "date": list(dates) * 4,  # repeat same dates for each id
            "is_blocked": [0] * 300 + [1] * 100,
        }
    )
    return df_test


def test_nullify_before_long_gaps(sample_df_large_gaps):

    df_result = nullify_before_long_gaps(
        df=sample_df_large_gaps,
        id_column="vin",
        date_column="date",
        measurement_columns=["x"],
        blocking_column="is_blocked",
        gap_cutoff_days=30,
    )

    # Assert no values removed from series with only small gaps
    assert len(df_result[(df_result["vin"] == "id1") & (df_result["x"].isna())]) == 20

    # Assert values removed from series before large gap in middle
    assert len(df_result[(df_result["vin"] == "id2") & (df_result["x"].isna())]) == 80

    # Assert values removed from series before large gap at end
    assert len(df_result[(df_result["vin"] == "id3") & (df_result["x"].isna())]) == 100

    # Assert no values removed from series with with blocking flag
    assert len(df_result[(df_result["vin"] == "id4") & (df_result["x"].isna())]) == 90


# DataFrame with outliers and edges
@pytest.fixture
def sample_df_outliers_and_edges():

    # Generate series with an outlier
    values_outlier = list(range(100))
    values_outlier[49] = 9999999

    # Generate series with an edge (this series should remain unaffected)
    values_edge = list(range(50)) + list(range(1000, 1050))

    # Combine series
    values = values_outlier + values_edge

    # Build date values for df
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")

    df_test = pd.DataFrame(
        {"vin": ["id1"] * 100 + ["id2"] * 100, "x": values, "date": list(dates) * 2}
    )
    return df_test


# Get config for hampel filter test
@pytest.fixture
def config_test_preprocess_hampel_filter():
    return load_config("config_tests.yaml")["preprocess"]["hampel_filter"]


def test_hampel_filter_forecasting_columns(
    sample_df_outliers_and_edges, config_test_preprocess_hampel_filter
):

    # Apply hampel filter
    df_result = hampel_filter_forecasting_columns(
        sample_df_outliers_and_edges,
        "vin",
        "date",
        config_test_preprocess_hampel_filter,
    )

    # Assert only the outlier is removed
    assert len(df_result[df_result["x"].isna()]) == 1
    assert df_result["x"].max() == 1049


# DataFrame with outliers
@pytest.fixture
def sample_df_outliers():

    # Generate series with an outlier
    values = list(range(100))
    values[49] = 9999999

    # Build date values for df
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")

    df_test = pd.DataFrame({"vin": ["id1"] * 100, "x": values, "date": list(dates)})
    return df_test


def test_nullify_decreased_measurements(sample_df_outliers):
    df_result = nullify_decreases(
        sample_df_outliers, id_column="vin", measurement_column="x", less_than_cutoff=0
    )

    # Assert outlier and preceding value have been set to null
    assert len(df_result[df_result["x"].isna()]) == 2
    assert df_result["x"].max() == 99


def test_nullify_all_values_before_decrease(sample_df_outliers):
    df_result = nullify_all_values_before_decreases(
        sample_df_outliers, id_column="vin", measurement_column="x", less_than_cutoff=0
    )

    # Assert outlier and all preceding values have been set to null
    assert len(df_result[df_result["x"].isna()]) == 51
    assert df_result["x"].max() == 99


# Get config for test_nullify_before_unrealistic_changes_decaying
@pytest.fixture
def config_test_nullify_before_unrealistic_changes_decaying():
    return load_config("config_tests.yaml")["preprocess"][
        "nullify_unrealistic_changes_decaying"
    ]


# DataFrame with decaying series containing unrealistic changes
@pytest.fixture
def sample_df_unrealistic_changes_decaying():

    # Generate series with large increase then decrease

    # Decaying series that reaches zero then "resets" to a high number
    values1 = list(range(50, 0, -1)) + list(range(1100, 1050, -1))

    # Decaying series with small unexpected increase
    values2 = list(range(50, 0, -1)) + list(range(50, 0, -1))

    # Decaying series with unrealically large decrease due to missing values
    values3 = list(range(500, 495, -1)) + [np.nan] * 90 + list(range(405, 400, -1))

    values = values1 + values2 + values3

    # Date column values
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")

    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 100 + ["id2"] * 100 + ["id3"] * 100,
            "x": values,
            "date": list(dates) * 3,
        }
    )
    return df_test


def test_nullify_before_unrealistic_changes_decaying(
    sample_df_unrealistic_changes_decaying,
    config_test_nullify_before_unrealistic_changes_decaying,
):

    # Nullify decaying each series before unrealistic changes
    df_result = nullify_before_unrealistic_changes(
        sample_df_unrealistic_changes_decaying,
        "vin",
        config_test_nullify_before_unrealistic_changes_decaying,
        mode="decaying",
    )

    # Assert no changes made to "reset"
    assert len(df_result[(df_result["vin"] == "id1") & (df_result["x"].isna())]) == 0

    # Assert values nulled before small increase
    assert len(df_result[(df_result["vin"] == "id2") & (df_result["x"].isna())]) == 50

    # Assert values nulled before large decrease
    assert len(df_result[(df_result["vin"] == "id3") & (df_result["x"].isna())]) == 95


# Get config for test_nullify_before_unrealistic_changes_rising
@pytest.fixture
def config_test_nullify_before_unrealistic_changes_rising():
    return load_config("config_tests.yaml")["preprocess"][
        "nullify_unrealistic_changes_rising"
    ]


# DataFrame with rising series containing unrealistic changes
@pytest.fixture
def sample_df_unrealistic_changes_rising():

    # Generate series with large increase then decrease

    # Decaying series that has large increase due to missing values in middle
    values = list(range(5)) + [np.nan] * 90 + list(range(95, 100))

    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")

    df_test = pd.DataFrame({"vin": ["id1"] * 100, "x": values, "date": list(dates)})
    return df_test


def test_nullify_before_unrealistic_changes_rising(
    sample_df_unrealistic_changes_rising,
    config_test_nullify_before_unrealistic_changes_rising,
):

    # Nullify decaying each series before unrealistic changes
    df_result = nullify_before_unrealistic_changes(
        sample_df_unrealistic_changes_rising,
        "vin",
        config_test_nullify_before_unrealistic_changes_rising,
        mode="rising",
    )

    # Assert values dropped before large increase after missing values
    assert len(df_result[df_result["x"].isna()]) == 95

    # Assert correct values were removed
    df_results_head = df_result.head(5)
    assert len(df_results_head[df_results_head["x"].isna()]) == 5
    df_results_tail = df_result.tail(5)
    assert len(df_results_tail[df_results_tail["x"].isna()]) == 0


# DataFrame with service resets
@pytest.fixture
def sample_df_with_service_reset():

    # Generate a decaying series with large increase similar to a TCU measurement reset due to service.
    values = list(range(49, -1, -1)) + list(range(10000, 9950, -1))

    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")

    df_test = pd.DataFrame({"vin": ["id1"] * 100, "x": values, "date": list(dates)})
    return df_test


def test_remove_resets_from_service_columns(sample_df_with_service_reset):

    df_result = remove_resets_from_service_columns(
        sample_df_with_service_reset, "vin", ["x"]
    )

    # Assert beginning of series has been increased beyond following values
    assert df_result["x"].max() == 10049
    assert df_result["x"].min() > 50
    assert len(df_result[df_result["x"].isna()]) == 0


# DataFrame with rising series containing missing x values and a is_booked flag
@pytest.fixture
def sample_df_missing_values_and_is_booked():

    # Generate a series with missing values
    values = list(range(25)) + [np.nan] * 50 + list(range(75, 100))

    # Generate is_booked column
    is_booked = [1] * 50 + [0] * 50

    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")

    df_test = pd.DataFrame(
        {"vin": ["id1"] * 100, "x": values, "is_booked": is_booked, "date": list(dates)}
    )
    return df_test


def test_interpolate_target_columns_when_booked(sample_df_missing_values_and_is_booked):

    df_result = interpolate_target_columns_when_booked(
        sample_df_missing_values_and_is_booked, "vin", "is_booked", ["x"]
    )

    # Assert all values where is_booked=1 or have been interpolated
    assert len(df_result[(df_result["x"].isna()) & (df_result["is_booked"] == 1)]) == 0

    # Assert all null values where is_booked=0 remain null
    assert len(df_result[(df_result["is_booked"] == 0) & (df_result["x"].isna())]) == 25


def test_forward_fill_missing(sample_df_missing_values_and_is_booked):

    df_result = forward_fill_missing(
        sample_df_missing_values_and_is_booked, "vin", ["x"]
    )

    # Assert all values have been forward filled
    assert len(df_result[df_result["x"].isna()]) == 0
    for i in range(25, 75):
        assert df_result["x"].iloc[i] == 24


# DataFrame with rising series containing missing x values and a is_booked flag
@pytest.fixture
def sample_df_groups_without_values():

    # Generate a series with missing values
    values = list(range(25)) + [np.nan] * 50

    # Generate dates
    dates = pd.date_range(start="2025-01-01", periods=25, freq="D")

    # Generate three groups
    vins = ["id1"] * 25 + ["id2"] * 25 + ["id3"] * 25

    df_test = pd.DataFrame({"vin": vins, "x": values, "date": list(dates) * 3})
    return df_test


def test_drop_cars_without_measurements(sample_df_groups_without_values):
    df_results = drop_cars_without_measurements(
        sample_df_groups_without_values, "vin", "date", ["x"]
    )

    # Assert groups with no values are dropped
    assert len(df_results) == 25
    assert len(df_results[df_results["x"].isna()]) == 0
