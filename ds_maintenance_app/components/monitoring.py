import pandas as pd
from typing import List
from copy import deepcopy


def _detect_series_resets(values: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """
    Detect resets in a decaying TCU measurement series.

    Counterpart to `_remove_series_resets`: rather than removing resets it
    records them. Any increase between consecutive non-null observations is
    treated as a manual reset (e.g. a service resetting
    gmx_service_oil_interval_percent back up after it has decayed).

    __PARAMETERS__
    values : pd.Series
        Series of numeric measurements.
    dates : pd.Series
        Dates aligned with `values`, used to report when each reset occurred.

    __RETURNS__
    pd.DataFrame
        One row per detected reset with columns:
        - "reset_date"   : date the reset was observed (date of the higher value)
        - "reset_amount" : amount the value increased by
    """

    # Keep only non-null observations, preserving order
    non_null = values.dropna()

    # Difference between consecutive non-null values (later - earlier)
    diffs = non_null.diff()

    # A reset is a positive change (an increase after decay)
    reset_mask = diffs > 0

    return pd.DataFrame(
        {
            "reset_date": dates.loc[diffs.index[reset_mask]].values,
            "reset_amount": diffs[reset_mask].values,
        }
    )


def detect_resets_in_service_columns(
    df: pd.DataFrame,
    id_column: str,
    date_column: str,
    reset_columns: List[str],
) -> pd.DataFrame:
    """
    Detect resets across multiple measurement columns.

    Counterpart to `remove_resets_from_service_columns`: instead of removing the
    resets it records them. Each series (grouped by `id_column`) is scanned for
    increases, which are treated as manual resets, e.g., after service.

    __PARAMETERS__
    df : pd.DataFrame
        Input DataFrame containing the measurement columns.
    id_column : str
        Name of the column used to group the DataFrame into series.
    date_column : str
        Name of the column containing dates, used to report when each reset occurred.
    reset_columns : List[str]
        List of measurement columns to scan for resets.

    __RETURNS__
    pd.DataFrame
        df_resets with one row per detected reset and columns:
        - id_column      : the series id (e.g. vin)
        - "service_type" : the measurement column the reset was found in
        - "reset_date"   : date the reset was observed
        - "reset_amount" : amount the value increased by
    """

    tmp_df = deepcopy(df)

    # Ensure each series is in chronological order before scanning for resets
    tmp_df = tmp_df.sort_values([id_column, date_column])

    resets = []
    for column in reset_columns:

        # Detect resets in each series grouped by id
        for series_id, group in tmp_df.groupby(id_column):
            df_series_resets = _detect_series_resets(group[column], group[date_column])

            # Record which series and service type each reset belongs to
            if not df_series_resets.empty:
                df_series_resets[id_column] = series_id
                df_series_resets["service_type"] = column
                resets.append(df_series_resets)

    # Combine all detected resets, or return an empty frame with the expected schema
    if resets:
        df_resets = pd.concat(resets, ignore_index=True)
        df_resets = df_resets[[id_column, "service_type", "reset_date", "reset_amount"]]
    else:
        df_resets = pd.DataFrame(
            columns=[id_column, "service_type", "reset_date", "reset_amount"]
        )

    return df_resets


def _detect_series_zero_crossings(values: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """
    Detect each time a decaying TCU measurement series crosses to zero or below.

    A crossing is recorded every time the value drops to zero or below after
    previously being above zero, so a series that runs down, resets above zero,
    then runs down again is recorded multiple times. Consecutive zero-or-below
    observations count as a single crossing.

    __PARAMETERS__
    values : pd.Series
        Series of numeric measurements.
    dates : pd.Series
        Dates aligned with `values`, used to report when each crossing occurred.

    __RETURNS__
    pd.DataFrame
        One row per crossing with the column:
        - "zero_date" : date the value crossed to zero (or below)
        Empty if the series never reaches zero.
    """

    # Keep only non-null observations, preserving order (NaNs are ignored)
    non_null = values.dropna()

    # Mark observations at or below zero
    zero_mask = non_null <= 0

    # A crossing starts where the value is <= 0 but the previous non-null
    # observation was above zero (a leading <= 0 counts as a crossing).
    episode_start = zero_mask & ~zero_mask.shift(1, fill_value=False)

    return pd.DataFrame(
        {"zero_date": dates.loc[episode_start.index[episode_start]].values}
    )


def detect_zero_crossings_in_service_columns(
    df: pd.DataFrame,
    id_column: str,
    date_column: str,
    zero_columns: List[str],
) -> pd.DataFrame:
    """
    Detect every time each series crosses to zero or below across multiple columns.

    Companion to `detect_resets_in_service_columns`: instead of recording resets
    it records each time a series (grouped by `id_column`) drops to zero or below
    after being above zero.

    __PARAMETERS__
    df : pd.DataFrame
        Input DataFrame containing the measurement columns.
    id_column : str
        Name of the column used to group the DataFrame into series.
    date_column : str
        Name of the column containing dates, used to report when each crossing occurred.
    zero_columns : List[str]
        List of measurement columns to scan for zero crossings.

    __RETURNS__
    pd.DataFrame
        df_zeros with one row per detected crossing and columns:
        - id_column      : the series id (e.g. vin)
        - "service_type" : the measurement column the crossing was found in
        - "zero_date"    : date the value crossed to zero (or below)
    """

    tmp_df = deepcopy(df)

    # Ensure each series is in chronological order before scanning for crossings
    tmp_df = tmp_df.sort_values([id_column, date_column])

    zeros = []
    for column in zero_columns:

        # Detect zero crossings in each series grouped by id
        for series_id, group in tmp_df.groupby(id_column):
            df_series_zero = _detect_series_zero_crossings(
                group[column], group[date_column]
            )

            # Record which series and service type each crossing belongs to
            if not df_series_zero.empty:
                df_series_zero[id_column] = series_id
                df_series_zero["service_type"] = column
                zeros.append(df_series_zero)

    # Combine all detected crossings, or return an empty frame with the expected schema
    if zeros:
        df_zeros = pd.concat(zeros, ignore_index=True)
        df_zeros = df_zeros[[id_column, "service_type", "zero_date"]]
    else:
        df_zeros = pd.DataFrame(columns=[id_column, "service_type", "zero_date"])

    return df_zeros
