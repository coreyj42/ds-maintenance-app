import pytest
import pandas as pd
import numpy as np

from ds_maintenance_app.components.monitoring import (
    detect_resets_in_service_columns,
    detect_zero_crossings_in_service_columns,
)


# DataFrame with decaying series containing service resets (increases)
@pytest.fixture
def sample_df_resets():
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 5 + ["id2"] * 5,
            # id1: one reset (10 -> 100, amount 90, on 2025-01-04)
            # id2: two resets (5 -> 60, amount 55; 40 -> 90, amount 50)
            "x": [50, 30, 10, 100, 80] + [20, 5, 60, 40, 90],
            "date": list(dates) * 2,
        }
    )
    return df_test


def test_detect_resets_in_service_columns(sample_df_resets):

    df_result = detect_resets_in_service_columns(sample_df_resets, "vin", "date", ["x"])

    # Assert expected schema and total number of resets
    assert list(df_result.columns) == [
        "vin",
        "service_type",
        "reset_date",
        "reset_amount",
    ]
    assert len(df_result) == 3

    # Assert id1 has a single reset with the correct amount and date
    id1 = df_result[df_result["vin"] == "id1"]
    assert len(id1) == 1
    assert id1["reset_amount"].iloc[0] == 90
    assert id1["reset_date"].iloc[0] == pd.Timestamp("2025-01-04")

    # Assert id2 has both resets with the correct amounts and dates
    id2 = df_result[df_result["vin"] == "id2"].sort_values("reset_date")
    assert list(id2["reset_amount"]) == [55, 50]
    assert list(id2["reset_date"]) == [
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-05"),
    ]


def test_detect_resets_handles_unsorted_input(sample_df_resets):

    # Reverse the rows so the input is not in chronological order
    shuffled = sample_df_resets.iloc[::-1].reset_index(drop=True)

    df_result = detect_resets_in_service_columns(shuffled, "vin", "date", ["x"])

    # Assert the same resets are detected after the function sorts internally
    assert len(df_result) == 3
    id2 = df_result[df_result["vin"] == "id2"].sort_values("reset_date")
    assert list(id2["reset_amount"]) == [55, 50]


# DataFrame with a reset across a gap of missing values
@pytest.fixture
def sample_df_resets_with_nan():
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 5,
            # Non-null sequence 30 -> 10 -> 50, so one reset of 40 on 2025-01-05
            "x": [30, np.nan, 10, np.nan, 50],
            "date": list(dates),
        }
    )
    return df_test


def test_detect_resets_ignores_nan(sample_df_resets_with_nan):

    df_result = detect_resets_in_service_columns(
        sample_df_resets_with_nan, "vin", "date", ["x"]
    )

    # Assert the reset is detected between consecutive non-null values
    assert len(df_result) == 1
    assert df_result["reset_amount"].iloc[0] == 40
    assert df_result["reset_date"].iloc[0] == pd.Timestamp("2025-01-05")


# Strictly decreasing series with no resets
@pytest.fixture
def sample_df_no_resets():
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    df_test = pd.DataFrame(
        {"vin": ["id1"] * 5, "x": [50, 40, 30, 20, 10], "date": list(dates)}
    )
    return df_test


def test_detect_resets_empty_when_no_increase(sample_df_no_resets):

    df_result = detect_resets_in_service_columns(
        sample_df_no_resets, "vin", "date", ["x"]
    )

    # Assert an empty frame with the expected schema is returned
    assert df_result.empty
    assert list(df_result.columns) == [
        "vin",
        "service_type",
        "reset_date",
        "reset_amount",
    ]


# DataFrame with series that cross to zero or below
@pytest.fixture
def sample_df_zero_crossings():

    # id1: crosses at the first 0 and again at -1 (consecutive <= 0 count once)
    df_id1 = pd.DataFrame(
        {
            "vin": ["id1"] * 9,
            "x": [5, 2, 0, 0, 5, 2, -1, -5, 5],
            "date": pd.date_range("2025-01-01", periods=9, freq="D"),
        }
    )

    # id2: never reaches zero
    df_id2 = pd.DataFrame(
        {
            "vin": ["id2"] * 5,
            "x": [10, 8, 6, 4, 2],
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
        }
    )

    # id3: starts <= 0 (leading crossing) then crosses again at -2
    df_id3 = pd.DataFrame(
        {
            "vin": ["id3"] * 3,
            "x": [-1, 5, -2],
            "date": pd.date_range("2025-01-01", periods=3, freq="D"),
        }
    )

    return pd.concat([df_id1, df_id2, df_id3], ignore_index=True)


def test_detect_zero_crossings_in_service_columns(sample_df_zero_crossings):

    df_result = detect_zero_crossings_in_service_columns(
        sample_df_zero_crossings, "vin", "date", ["x"]
    )

    # Assert expected schema and total number of crossings
    assert list(df_result.columns) == ["vin", "service_type", "zero_date"]
    assert len(df_result) == 4

    # Assert id1 crosses twice on the expected dates
    id1 = df_result[df_result["vin"] == "id1"].sort_values("zero_date")
    assert list(id1["zero_date"]) == [
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-07"),
    ]

    # Assert id2 never crosses
    assert len(df_result[df_result["vin"] == "id2"]) == 0

    # Assert id3 records the leading crossing and the later crossing
    id3 = df_result[df_result["vin"] == "id3"].sort_values("zero_date")
    assert list(id3["zero_date"]) == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-03"),
    ]


# DataFrame with a zero crossing across a gap of missing values
@pytest.fixture
def sample_df_zero_crossings_with_nan():
    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 5,
            # Non-null sequence 5 -> -1 -> 5 -> -2, so two crossings
            "x": [5, np.nan, -1, 5, -2],
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
        }
    )
    return df_test


def test_detect_zero_crossings_ignores_nan(sample_df_zero_crossings_with_nan):

    df_result = detect_zero_crossings_in_service_columns(
        sample_df_zero_crossings_with_nan, "vin", "date", ["x"]
    )

    # Assert both crossings are detected between consecutive non-null values
    assert len(df_result) == 2
    assert list(df_result.sort_values("zero_date")["zero_date"]) == [
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-05"),
    ]


# DataFrame with crossings in two separate columns
@pytest.fixture
def sample_df_multi_column():
    df_test = pd.DataFrame(
        {
            "vin": ["id1"] * 3,
            "x1": [5, -1, 5],  # crosses on 2025-01-02
            "x2": [-2, 3, 4],  # leading crossing on 2025-01-01
            "date": pd.date_range("2025-01-01", periods=3, freq="D"),
        }
    )
    return df_test


def test_detect_zero_crossings_multiple_columns(sample_df_multi_column):

    df_result = detect_zero_crossings_in_service_columns(
        sample_df_multi_column, "vin", "date", ["x1", "x2"]
    )

    # Assert both columns are scanned and labelled
    assert set(df_result["service_type"]) == {"x1", "x2"}
    assert len(df_result) == 2

    x1 = df_result[df_result["service_type"] == "x1"]
    assert list(x1["zero_date"]) == [pd.Timestamp("2025-01-02")]

    x2 = df_result[df_result["service_type"] == "x2"]
    assert list(x2["zero_date"]) == [pd.Timestamp("2025-01-01")]
