import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, List
from copy import deepcopy

from ds_package_anomaly_detection.hampel import HampelAnomalyDetection


def nullify_zeros(df: pd.DataFrame, measurement_columns: Union[List[str], np.array]):
    """
    Change all zero values in columns to null

    __PARAMETERS__
    df: pd.DataFrame
        DataFrame of TCU measurements time-series
    measurement_columns: Union[List[str],np.array]
        list of columns to check

    __RETURNS__
    pd.DataFrame
        An edited DataFrame
    """
    df_tmp = deepcopy(df)

    df_tmp[measurement_columns] = df_tmp[measurement_columns].mask(
        df_tmp[measurement_columns] == 0
    )
    return df_tmp


def nullify_if_other_not_null(
    df: pd.DataFrame, id_column: str, target_column: str, reference_column: str
) -> pd.DataFrame:
    """
    Change all values in target_column to null if a value exists in the same row in reference_column.

    __PARAMETERS__
    df: pd.DataFrame
        DataFrame of TCU measurements time-series
    id_column: str
        Name of column containing the series ids
    target_column:
        Name of the column to apply the nullification to
    reference_column:
        Name of the column to check for non-null values

    __RETURNS__
    pd.DataFrame
        An edited DataFrame
    """
    df_tmp = deepcopy(df)

    # Create mask of vins with reference_column present
    mask = df_tmp[reference_column].notna().groupby(df_tmp[id_column]).transform("any")

    # Set target to null if reference present
    df_tmp.loc[mask, target_column] = np.nan

    return df_tmp


def convert_booleans(df):
    """
    Changes all values in Boolean columns to 1 and 0 instead of True and False

    __PARAMETERS__
    df: pd.DataFrame
        DataFrame of TCU measurements time-series

    __RETURNS__
    pd.DataFrame
        An edited DataFrame
    """
    try:
        for col in df.columns:
            if col.startswith("is_"):
                df[col] = df[col].astype(int)
        return df
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert boolean columns to int: {e}")


def split_future_data(
    df: pd.DataFrame,
    id_column: str,
    date_column: str,
    dynamic_features: Union[List[str], np.array],
) -> tuple:
    """
    Split a DataFrame into past and future records based on today's date.

    __PARAMETERS__
    df: pd.DataFrame
        DataFrame of TCU measurements time-series
    id_column: str
        Name of column containing the series ids
    date_column: str
        Name of the column containing dates used for the split.
    dynamic_features: Union[List[str],np.array]
        Dynamic features required by future table in forecast

    __RETURNS__
    tuple[pd.DataFrame, pd.DataFrame]
        df_past   : Rows with dates <= today
        df_future : Rows with dates > today
    """
    # Get current date
    today = pd.Timestamp.today().normalize()

    # Step 2: Split into past vs future
    df_past = deepcopy(df[df[date_column] <= today])
    df_future = deepcopy(
        df[df[date_column] > today][[id_column] + [date_column] + dynamic_features]
    )

    return df_past, df_future


def nullify_before_long_gaps(
    df: pd.DataFrame,
    id_column: str,
    date_column: str,
    measurement_columns: Union[List[str], np.array],
    blocking_column: str,
    gap_cutoff_days: int,
):
    """
    Look for large streaks of null values (of length gap_cutoff_days or more) in each series. If found set all values
    before the first date in the streak to null.
    This is done to prevent values before a large period of missing values being used in forecasting.
    Any values where blocking_column == 1 are ignored when counting the streak of null values.

    __PARAMETERS__
    df: pd.DataFrame
        DataFrame of TCU measurements time-series
    id_column: str
        Name of column containing the series ids
    date_column: str
        Name of column containing dates
    measurement_columns: Union[List[str],np.array]
        list of columns to apply filter to
    blocking_column: str
        Name of column containing blocking flag. If blocking_column == 1. The day
        is ignored when counting streaks of null values.
    gap_cutoff_days: int
        Minimum number of consecutive days with null values (excluding rows where
        blocking_column == 1) required to define a streak of missing values large enough
        to have all previous values removed. If a gap of at least
        this length is detected for a given id, all measurement values before the streak
        are set to null.

    __RETURNS__
    pd.DataFrame
        An edited DataFrame
    """

    df_no_gaps = deepcopy(df)

    # Ensure correct ordering
    df_no_gaps = df_no_gaps.sort_values([id_column, date_column])

    # Loop over each measurement column
    for col in measurement_columns:

        # Create null indicator
        df_no_gaps["is_null"] = df_no_gaps[col].isna() & (
            df_no_gaps[blocking_column] == 0
        )

        # Identify consecutive runs per id
        df_no_gaps["null_run_id"] = (
            df_no_gaps["is_null"]
            .ne(df_no_gaps.groupby(id_column)["is_null"].shift())
            .groupby(df_no_gaps["vin"])
            .cumsum()
        )

        # Length of each run
        df_no_gaps["null_run_length"] = df_no_gaps.groupby([id_column, "null_run_id"])[
            "is_null"
        ].transform("size")

        # Get end date of last large gap per vin
        latest_large_gap_dates = (
            df_no_gaps.loc[
                df_no_gaps["is_null"]
                & (df_no_gaps["null_run_length"] >= gap_cutoff_days),
                [id_column, date_column],
            ]
            .groupby(id_column, as_index=False)
            .last()
        )

        # Merge cutoff date
        df_no_gaps = df_no_gaps.merge(
            latest_large_gap_dates.rename(columns={date_column: "cutoff_date"}),
            on=id_column,
            how="left",
        )

        # Null all values before last large gap
        df_no_gaps.loc[df_no_gaps[date_column] < df_no_gaps["cutoff_date"], col] = (
            np.nan
        )

        # Drop helper columns before next iteration
        df_no_gaps = df_no_gaps.drop(
            columns=["is_null", "null_run_id", "null_run_length", "cutoff_date"]
        )

    return df_no_gaps


def hampel_filter_forecasting_columns(
    df_features: pd.DataFrame,
    id_column: str,
    date_column: str,
    hampel_config_dict: dict,
):
    """
    Apply a Hampel filter to selected columns of a time-series DataFrame to detect
    and replace outliers prior to forecasting.

    The filter is applied independently per series (defined by `id_column`)
    and per measurement column using the configuration provided.

    __PARAMETERS__
    df: pd.DataFrame
        DataFrame of TCU measurements time-series
    id_column: str
        Name of column containing the series ids
    date_column: str
        Name of column containing dates
    hampel_config_dict : dict
        Dictionary mapping column names to Hampel filter parameters.
        Example structure:

        {
            "measurement_col": {
                "neighbours": int,   # number of points on each side of the window (k)
                "threshold": float   # threshold multiplier (t)
            }
        }

    __RETURNS__
    pd.DataFrame
        An edited DataFrame
    """

    # Apply Hampel filter to each column
    for measurement_column in hampel_config_dict:

        # Extract columns needed for anomaly detection
        df_tmp = deepcopy(df_features[[date_column, id_column, measurement_column]])

        # Initialise hampel anomaly detector
        hampel_ad = HampelAnomalyDetection(
            num_neighbours=hampel_config_dict[measurement_column]["neighbours"],
            threshold=hampel_config_dict[measurement_column]["threshold"],
            date=date_column,
            id=id_column,
            target=measurement_column,
        )

        # Get anomalies
        df_anomalies = hampel_ad.get_anomalies(df_tmp)

        # Set anomalies to NaN
        df_features.loc[
            df_tmp.index[df_anomalies["is_anomaly"]], measurement_column
        ] = np.nan

    return df_features


# check for decreases in odometer values
def filter_decrease_in_odometer(df):
    # check order
    tmp_diffs = deepcopy(df[["date_updated", "vin", "gmx_odometer"]])
    tmp_diffs = tmp_diffs.sort_values(["vin", "date_updated"])

    # calculate diffs
    tmp_diffs["decrease"] = tmp_diffs.groupby("vin")["gmx_odometer"].transform(
        lambda s: s.ffill().diff()
    )

    # filter decreasing diffs
    tmp_diffs = (
        tmp_diffs[tmp_diffs["decrease"] < 0]
        .sort_values("decrease")
        .reset_index(drop=True)
    )

    return tmp_diffs


def _nullify_decreased_values(series: pd.Series, less_than_cutoff: int):
    """
    Check a series differences for decreases. If a decrease is found that is less than the threshold, it and the preceding value are set to null.

    __PARAMETERS__
    series: pd.Series
        series to be checked and filtered
    less_than_cutoff: int
        Threshold for decreases. Differences smaller (more negative) than this
        cutoff are considered significant decreases.

    __RETURNS__
    pd.Series
        An edited series
    """

    # Ensure less_than_cutoff is negative
    less_than_cutoff = -abs(less_than_cutoff)

    tmp_series = deepcopy(series)

    # Forward fill only for comparison
    tmp_series_ffilled = tmp_series.ffill()

    # Compute differences
    diffs = tmp_series_ffilled.diff()

    # Indices where a decrease happens
    decrease_idx = diffs[diffs < less_than_cutoff].index

    # Mask to null current and previous non-null value
    mask = pd.Series(False, index=tmp_series.index)

    for i in decrease_idx:
        # Null current
        mask.loc[i] = True
        # Find previous non-null index and null it too
        prev_idx = tmp_series.loc[: i - 1].last_valid_index()
        if prev_idx is not None:
            mask.loc[prev_idx] = True

    # Apply mask
    tmp_series.loc[mask] = pd.NA

    return tmp_series


def nullify_decreases(
    df: pd.DataFrame,
    id_column: str,
    measurement_column: str,
    less_than_cutoff: Union[int, float],
):
    """
    Nullify decreases in a measurement column for each series in a DataFrame.

    For the specified `measurement_column`, the function checks consecutive values
    within each series (grouped by `id_column`). If a difference less than the
    specified `less_than_cutoff` is detected, both the decreased value and the
    preceding valid value are set to null. This is useful for cleaning odometer or
    cumulative measurement series where decreases are not expected.

    This step is performed before nulling all values before a decrease in order to
    clean any spikes that were missed by other preprocessing steps like the Hampel
    filter before removing all values before the decrease.

    __PARAMETERS__
    df: pd.DataFrame
        DataFrame containing measurement column to be filtered
    id_column: str
        Name of column containing the series ids
    measurement_column: str
        Name of column to apply the filter to
    less_than_cutoff: Union[int, float]
        Threshold for decreases. Differences smaller (more negative) than this
        cutoff are considered significant decreases.

    __RETURNS__
    pd.Series
        An edited series
    """
    tmp_df = deepcopy(df)

    # Apply _nullify_decreased_values per series
    tmp_df[measurement_column] = tmp_df.groupby(id_column)[
        measurement_column
    ].transform(lambda s: _nullify_decreased_values(s, less_than_cutoff))

    return tmp_df


def _null_all_before_decreased_values(
    series: pd.Series, less_than_cutoff: Union[int, float]
) -> pd.Series:
    """
    Nullify all values in a series up to the last detected decrease exceeding a threshold.

    This function scans a pandas Series for decreases between consecutive values.
    Any decrease smaller than `less_than_cutoff` (i.e., more negative) is considered
    a significant decrease. All values up to and including the last such decrease
    are set to null (`pd.NA`). Useful for cumulative measurements or
    odometer-like series where decreases indicate invalid data.

    Applied after all other outlier removal steps as a more aggressive filtering method
    to remove all values before missed irregular measurements.

    __PARAMETERS__
    series : pd.Series
        The series to check for decreases and apply nullification.
    less_than_cutoff : int or float
        Threshold for decreases. Differences smaller (more negative) than this
        cutoff are considered significant decreases.

    __RETURNS__
    pd.Series
        A copy of the input series with all values before (and including) the last
        significant decrease set to null (`pd.NA`).
    """

    # Ensure less_than_cutoff is negative
    less_than_cutoff = -abs(less_than_cutoff)

    tmp_series = deepcopy(series)

    # Forward-fill for comparison
    tmp_series_ffilled = tmp_series.ffill()

    # Detect decreases
    decreases = tmp_series_ffilled.diff().lt(less_than_cutoff)

    if not decreases.any():
        return tmp_series  # nothing to null

    # Get index of last decrease
    last_idx = decreases[decreases].index[-1]

    # Null everything up to last decrease
    tmp_series_nulled = deepcopy(tmp_series)
    tmp_series_nulled.loc[tmp_series_nulled.index <= last_idx] = pd.NA

    return tmp_series_nulled


def nullify_all_values_before_decreases(
    df: pd.DataFrame,
    id_column: str,
    measurement_column: str,
    less_than_cutoff: Union[int, float],
):
    """
    Nullify all values in a measurement column up to the last significant decrease
    for each series in a DataFrame.

    For the specified `measurement_column`, the function checks consecutive values
    within each series (grouped by `id_column`). If a decrease smaller than the
    specified `less_than_cutoff` is detected, all values up to and including
    the last such decrease are set to null (`pd.NA`). This is useful for cumulative
    measurements or odometer-like series where decreases indicate invalid data.

    __PARAMETERS__
    df : pd.DataFrame
        Input DataFrame containing the measurement column to be filtered.
    id_column : str
        Name of the column used to group the DataFrame into series.
    measurement_column : str
        Name of the measurement column to apply the nullification to.
    less_than_cutoff : int or float
        Threshold for decreases. Differences smaller (more negative) than this
        cutoff are considered significant decreases.

    __RETURNS__
    pd.DataFrame
        A copy of the input DataFrame with all values in the specified
        measurement column nullified up to the last significant decrease for
        each series.
    """
    tmp_df = deepcopy(df)

    # Apply _nullify_decreased_values per series
    tmp_df[measurement_column] = tmp_df.groupby(id_column)[
        measurement_column
    ].transform(lambda s: _null_all_before_decreased_values(s, less_than_cutoff))

    return tmp_df


def nullify_before_unrealistic_changes(
    df: pd.DataFrame,
    id_column: str,
    change_cutoffs_dict: Dict[
        str, Union[float, Tuple[float, float]]
    ],  # Depends on mode.
    mode: str = "rising",  # "rising" | "decaying"
) -> pd.DataFrame:
    """
    Nullify values occurring before the last unrealistic change in a measurement series.

    For each column specified in `change_cutoffs_dict`, forward-filled differences are
    computed within each series (grouped by `id_column`). Rows occurring before the
    final detected unrealistic change are set to null (`np.nan`) for the corresponding
    column.

    Two modes are supported with different definitions of an unrealistic change:

    mode="rising":
        A value is considered unrealistic if it represents a positive increase larger
        than the provided cutoff.

        This mode is intended for monotonically increasing measurements such as
        odometer-like series, where large jumps typically indicate resets, data errors,
        or preprocessing artefacts.

    mode="decaying":
        A value is considered unrealistic if it represents either:
        - a large negative decrease smaller than the provided negative cutoff, or
        - a small positive increase below the provided positive cutoff.

        This mode is intended for decaying service-interval style measurements (e.g.
        TCU columns) that gradually decrease over time and then increase sharply when
        a service is performed. Small increases are treated as noise rather than true
        resets.

    Nullifying early values is useful when the initial portion of a time-series may be
    unreliable due to resets, sensor errors, or incomplete historical capture.

    __PARAMETERS__
    df : pd.DataFrame
        Input DataFrame containing the identifier column and measurement columns.

    id_column : str
        Column used to group the DataFrame into independent series.

    change_cutoffs_dict : dict
        Dictionary mapping column names to cutoff values.
        - mode="rising": float cutoff per column.
        - mode="decaying": tuple of (negative_cutoff, positive_cutoff) per column.

    mode : {"rising", "decaying"}, default "rising"
        Determines the rule used to identify unrealistic changes.

    __RETURNS__
    pd.DataFrame
        Copy of the input DataFrame with values nullified up to the last detected
        unrealistic change for each series and column.
    """

    tmp_df = deepcopy(df)
    tmp_df["_pos"] = tmp_df.groupby(id_column).cumcount()

    for col, cutoff in change_cutoffs_dict.items():

        # diff
        ffilled = tmp_df.groupby(id_column)[col].ffill()
        diff = ffilled.groupby(tmp_df[id_column]).diff()

        if mode == "rising":
            flag = diff > cutoff
        elif mode == "decaying":
            neg_cut, pos_cut = cutoff
            flag = (diff < neg_cut) | ((diff > 0) & (diff < pos_cut))

        else:
            raise ValueError("mode must be 'rising' or 'decaying'")

        last_bad_pos = tmp_df.where(flag).groupby(id_column)["_pos"].max()
        tmp_df["_last_bad_pos"] = tmp_df[id_column].map(last_bad_pos)

        mask = tmp_df["_pos"] < tmp_df["_last_bad_pos"]

        tmp_df.loc[mask, col] = np.nan

        tmp_df.drop(columns="_last_bad_pos", inplace=True)

    tmp_df.drop(columns="_pos", inplace=True)

    return tmp_df


def _remove_series_resets(values: pd.Series) -> pd.Series:
    """
    Remove resets in a decaying TCU measurement series.

    Any increase in the series is treated as a manual reset, and previous
    values are adjusted cumulatively to remove the effect of resets.

    These resets occur due to services being carried out on vans when the measurement
    reaches zero. For example in gmx_service_oil_interval_percent will start at 100 and decay
    as the car is driven. When it reaches zero the car will be blocked for a service. When the
    service is performed the values resets to 100. The purpose of removing these resets is to
    better model the rate of decay in the series without the sudden increases affecting the
    model.

    __PARAMETERS__
    values : pd.Series
        Series of numeric measurements.

    __RETURNS__
    pd.Series
        Series with resets removed, same index as input.
    """

    # If the series has no valid values, return as-is
    non_null = deepcopy(values).dropna()
    if non_null.empty:
        return values

    # Initialise variables
    value_to_add = 0  # tracks total step size
    cum_values = np.empty(len(values))
    cum_values[:] = np.nan

    # Initialise previous value with final non-null value in series. Done here in case last actual value is null
    # Records previous value in loop to avoid effects of nans
    previous_value = values.dropna().iloc[-1]

    # Loop backwards over values
    for i in range(len(values) - 1, -1, -1):
        current_value = values.iloc[i]

        # Add the final value in series to beginning of reversed series
        if i == len(values) - 1:
            cum_values[i] = current_value

        else:
            # If a reset occurred, add difference in values to value_to_add
            if previous_value > current_value:
                value_to_add += previous_value - current_value

            # Update value with cumulative applied reset amount
            cum_values[i] = current_value + value_to_add

        if not pd.isna(current_value):
            previous_value = current_value

    return pd.Series(cum_values, index=values.index)


def remove_resets_from_service_columns(
    df: pd.DataFrame, id_column, remove_resets_columns: List[str]
) -> pd.DataFrame:
    """
    Apply reset removal to multiple measurement columns in a DataFrame.

    Each series (grouped by `id_column`) is passed to `_remove_series_resets`
    to remove increases that are likely due to manual resets, e.g., after
    service.

    __PARAMETERS__
    df : pd.DataFrame
        Input DataFrame containing the measurement columns.
    id_column : str
        Name of the column used to group the DataFrame into series.
    remove_resets_columns : List[str]
        List of measurement columns to apply reset removal to.

    __RETURNS__
    pd.DataFrame
        Copy of input DataFrame with resets removed in specified columns.
    """

    tmp_df = deepcopy(df)

    for value in remove_resets_columns:

        # Apply _remove_series_resets to each series grouped by id
        tmp_df[value] = tmp_df.groupby(id_column)[value].transform(
            _remove_series_resets
        )

    return tmp_df


def _interpolate_booked_series(values: pd.Series, booked: pd.Series) -> pd.Series:
    """
    Interpolate values only where booked == 1 or value is non-null.

    __PARAMETERS__
    values : pd.Series
        Series of numeric measurements to interpolate.
    booked : pd.Series
        Boolean or 0/1 series indicating if a value is booked (1) or not (0).

    __RETURNS__
    pd.Series
        Interpolated series, aligned with input index.
    """

    # Copy to avoid modifying original
    tmp_values = deepcopy(values)

    # Mask: interpolate only where booked or value exists
    mask = (booked == 1) | (values.notna())

    # Interpolate along the masked entries
    tmp_values[mask] = tmp_values[mask].interpolate(method="linear")

    return tmp_values


def interpolate_target_columns_when_booked(
    df: pd.DataFrame, id_column: str, booked_column: str, interpolate_columns: List[str]
) -> pd.DataFrame:
    """
    Interpolate multiple columns in a DataFrame only for rows
    that are booked or already contain values.

    Uses `.transform` to ensure the output aligns perfectly with the original index,
    avoiding MultiIndex issues from `.apply`.

    __PARAMETERS__
    df : pd.DataFrame
        Input DataFrame containing measurement columns and a booking indicator.
    id_column : str
        Column used to group the DataFrame into independent series.
    booked_column : str
        Column indicating whether a value is booked (1 = booked).
    interpolate_columns : List[str]
        List of column names to interpolate.

    __RETURNS__
    pd.DataFrame
        Copy of DataFrame with specified columns interpolated only
        for booked or non-null rows.
    """
    tmp_df = deepcopy(df)

    # Apply _interpolate_booked_series per column within each group
    for col in interpolate_columns:
        tmp_df[col] = tmp_df.groupby(id_column)[col].transform(
            lambda s: _interpolate_booked_series(s, tmp_df.loc[s.index, booked_column])
        )

    return tmp_df


def forward_fill_missing(df: pd.DataFrame, id_column: str, ffill_columns: List[str]):
    """
    Forward-fill missing values within each group defined by `id_column`
    for the specified `ffill_columns`.

    __PARAMETERS__
    df : pd.DataFrame
        Input DataFrame.
    id_column : str
        Column used to define groups (e.g., entity ID in a time-series).
    ffill_columns : List[str]
        Columns on which forward fill should be applied.

    __RETURNS__
    pd.DataFrame
        DataFrame with forward-filled values and cleaned rows.
    """
    tmp_df = deepcopy(df)

    # Forward fill each group
    tmp_df[ffill_columns] = tmp_df.groupby(id_column)[ffill_columns].ffill()

    # Drop missing rows with all missing values at head of each series
    tmp_df.dropna(subset=ffill_columns, how="all", inplace=True)
    tmp_df.reset_index(drop=True, inplace=True)

    return tmp_df


def drop_cars_without_measurements(
    df: pd.DataFrame, id_column: str, date_column: str, service_columns: List[str]
) -> pd.DataFrame:
    """
    Remove groups (e.g., vins) that do not contain any valid measurement
    values across the specified service columns.

    A group is considered valid if it has at least one non-null value in any
    of the provided `service_columns`.

    The resulting DataFrame is filtered, sorted by `id_column` and
    `date_column`, and returned with a clean integer index.

    __PARAMETERS__
    df : pd.DataFrame
        Input DataFrame.
    id_column : str
        Column identifying the entity (e.g., VIN).
    date_column : str
        Column used for chronological ordering.
    service_columns : List[str]
        Columns containing measurement/service values to validate.

    __RETURNS__
    pd.DataFrame
        Filtered DataFrame containing only entities with at least one
        valid measurement.
    """
    tmp_df = deepcopy(df)

    # Identify vins that have at least one non-null value in service_columns
    valid_car_ids = tmp_df.groupby(id_column)[service_columns].apply(
        lambda x: x.notna().any().any()
    )
    valid_car_ids = valid_car_ids[valid_car_ids].index

    # Keep only valid cars
    tmp_df = (
        tmp_df[tmp_df[id_column].isin(valid_car_ids)]
        .sort_values([id_column, date_column])
        .reset_index(drop=True)
    )

    return tmp_df
