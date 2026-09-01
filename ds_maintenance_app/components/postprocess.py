from typing import Dict, List, Optional
import pandas as pd
from copy import deepcopy


def _merge_future_features(
    df_forecast: pd.DataFrame,
    X_future: pd.DataFrame,
    dynamic_features: List[str],
) -> pd.DataFrame:
    """
    Merge dynamic future features into forecast results.

    This step enriches the forecast output with future dynamic features needed for postprocessing

    __PARAMETERS__
    df_forecast : pd.DataFrame
        Forecast DataFrame containing at minimum:
        - unique_id
        - ds (timestamp)
    X_future : pd.DataFrame
        Future feature DataFrame containing dynamic features aligned on
        unique_id and ds.
    dynamic_features : list[str]
        List of column names representing time-varying features to merge.

    __RETURNS__
    pd.DataFrame
        Forecast DataFrame with dynamic features merged.
    """

    return df_forecast.merge(
        X_future[["unique_id", "ds"] + dynamic_features],
        how="left",
        on=["unique_id", "ds"],
    )


def _fill_missing_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing confidence interval bounds with point forecasts.

    If either the lower or upper bound is missing, both bounds are set
    equal to the point forecast to ensure numerical consistency.

    The missing values are caused by the series having too few values for
    confidence interval calculations.

    __PARAMETERS__
    df : pd.DataFrame
        Forecast DataFrame containing columns:
        - yhat
        - low
        - high

    __RETURNS__
    pd.DataFrame
        DataFrame with missing interval bounds filled.
    """

    tmp_df = deepcopy(df)

    mask = tmp_df["low"].isna()

    tmp_df.loc[mask, "low"] = tmp_df.loc[mask, "yhat"]
    tmp_df.loc[mask, "high"] = tmp_df.loc[mask, "yhat"]

    return tmp_df


def _apply_conditional_flattening(
    df: pd.DataFrame,
    postprocess_dict: Dict[str, int],
) -> pd.DataFrame:
    """
    Flatten forecasts to zero based on a feature/value rule.

    Any rows matching the specified condition will have their forecast
    deltas set to zero for yhat, low, and high.

    __PARAMETERS__
    df : pd.DataFrame
        Forecast DataFrame containing forecast columns and the feature
        used for conditional logic.
    postprocess_dict : dict[str, Any]
        Dictionary defining a single conditional rule.
        Example:
            {"is_booked": 0}

    __RETURNS__
    pd.DataFrame
        DataFrame with conditional flattening applied.
    """

    tmp_df = deepcopy(df)

    # Extract the feature and value for which target should be set to zero
    feature = list(postprocess_dict.keys())[0]
    value = list(postprocess_dict.values())[0]

    # Apply flattening to each forecast with check in case CI not used
    for col in ["yhat", "low", "high"]:
        if col in tmp_df.columns:
            tmp_df.loc[tmp_df[feature] == value, col] = 0.0

    return tmp_df


def _clip_to_trend(
    df: pd.DataFrame,
    trend: Optional[str],
) -> pd.DataFrame:
    """
    Remove unrealistic values that move against the expected trend direction.
    For example gmx_odometer values are expected to always increase or remain the same.
    Forecast deltas are clipped to enforce monotonic direction constraints.

    __PARAMETERS__
    df : pd.DataFrame
        Forecast DataFrame containing columns:
        - yhat
        - low
        - high
    trend : str | None
        Expected direction of the target series:
        - "up"   → values clipped to be non-negative
        - "down" → values clipped to be non-positive
        - None   → no clipping applied

    __RETURNS__
    pd.DataFrame
        DataFrame with trend clipping applied.
    """

    tmp_df = deepcopy(df)

    if trend == "up":
        tmp_df[["yhat", "low", "high"]] = tmp_df[["yhat", "low", "high"]].clip(lower=0)

    elif trend == "down":
        tmp_df[["yhat", "low", "high"]] = tmp_df[["yhat", "low", "high"]].clip(upper=0)

    return tmp_df


def _merge_last_observation(
    df: pd.DataFrame,
    df_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the last observed level for each series.

    The last historical value is required to reconstruct cumulative
    forecasts from predicted deltas.

    __PARAMETERS__
    df : pd.DataFrame
        Forecast DataFrame containing unique_id.
    df_stats : pd.DataFrame
        Historical statistics DataFrame containing:
        - unique_id
        - y (observed level)

    __RETURNS__
    pd.DataFrame
        DataFrame with the last observed value merged as column `y_last`.
    """

    last_y = (
        df_stats.groupby("unique_id")
        .tail(1)[["unique_id", "y"]]
        .rename(columns={"y": "y_last"})
    )

    return df.merge(last_y, on="unique_id", how="left")


def _reconstruct_full_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct cumulative forecast levels from predicted deltas.

    The function converts delta predictions into absolute levels using
    the last observed value for each series. Original delta values are
    preserved in new columns with the suffix `_delta`.

    __PARAMETERS__
    df : pd.DataFrame
        Forecast DataFrame containing:
        - unique_id
        - yhat, low, high (delta predictions)
        - y_last (last observed level)

    __RETURNS__
    pd.DataFrame
        DataFrame with reconstructed cumulative forecasts and preserved
        delta columns.
    """

    tmp_df = deepcopy(df)

    # Columns to group by for cumsum
    group_cols = ["unique_id"]

    # Check if cutoff present due to backtesting
    if "cutoff" in tmp_df.columns:
        group_cols.append("cutoff")

    for col in ["yhat", "low", "high"]:

        # Check if column exists to handle CI missing
        if col in tmp_df.columns:

            tmp_df[f"{col}_delta"] = tmp_df[col]

            tmp_df[col] = tmp_df.groupby(group_cols)[col].cumsum() + tmp_df["y_last"]

    return tmp_df


def postprocess_forecast(
    df_forecast_results: pd.DataFrame,
    X_future: pd.DataFrame,
    df_stats: pd.DataFrame,
    dynamic_features: List[str],
    postprocess_dict: Dict[str, int],
    trend: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply the full post-processing pipeline to forecast results.

    The pipeline includes feature merging, confidence interval correction,
    conditional flattening, trend clipping, and reconstruction of cumulative
    forecast levels.

    __PARAMETERS__
    df_forecast_results : pd.DataFrame
        Raw forecast output.
    X_future : pd.DataFrame
        Future feature DataFrame.
    df_stats : pd.DataFrame
        Historical statistics containing last observed values.
    dynamic_features : list[str]
        Dynamic features to merge.
    postprocess_dict : dict[str, Any]
        Conditional flattening rule.
    trend : str | None, default=None
        Expected trend direction.

    __RETURNS__
    pd.DataFrame
        Fully post-processed forecast DataFrame.
    """

    tmp_df = deepcopy(df_forecast_results)

    # Apply each postprocessing step
    tmp_df = _merge_future_features(tmp_df, X_future, dynamic_features)
    tmp_df = _fill_missing_intervals(tmp_df)
    tmp_df = _apply_conditional_flattening(tmp_df, postprocess_dict)
    tmp_df = _clip_to_trend(tmp_df, trend)
    tmp_df = _merge_last_observation(tmp_df, df_stats)
    tmp_df = _reconstruct_full_forecast(tmp_df)

    return tmp_df


def postprocess_backtest(
    df_backtest_results: pd.DataFrame,
    df_stats: pd.DataFrame,
    dynamic_features: List[str],
    postprocess_dict: Dict[str, int],
) -> pd.DataFrame:
    """
    Apply post-processing to backtest results.

    This function applies a simplified post-processing pipeline for backtest outputs.
    The steps include:
    - Merging conditional features for flattening
    - Applying conditional flattening rules
    - Merging original y-values at the cutoff points
    - Reconstructing cumulative forecasts from deltas

    __PARAMETERS__
    df_backtest_results : pd.DataFrame
        Raw backtest forecast results containing at least:
        - unique_id
        - yhat, low, high (forecasted deltas)
        - cutoff (timestamp of cutoff for backtest)
    df_stats : pd.DataFrame
        Historical statistics DataFrame containing:
        - unique_id
        - y (observed values)
        - ds (date column)
    dynamic_features : list[str]
        List of column names representing time-varying features to merge.
    postprocess_dict : dict[str, int]
        Conditional flattening rule applied to forecast results.
        Example: {"is_booked": 0}

    __RETURNS__
    pd.DataFrame
        Backtest results DataFrame with post-processing applied,
        including reconstructed cumulative forecasts and cutoff values.
    """

    tmp_df = deepcopy(df_backtest_results)

    # Apply each step of postprocess
    tmp_df = _merge_future_features(tmp_df, df_stats, dynamic_features)
    tmp_df = _apply_conditional_flattening(tmp_df, postprocess_dict)

    # Get actual y values
    df_y = deepcopy(df_stats[["unique_id", "ds", "y"]])

    # Merge y
    tmp_df = tmp_df.merge(df_y, on=["unique_id", "ds"], how="left")

    # Get cutoff y values
    df_y_cutoff = deepcopy(df_stats[["unique_id", "ds", "y"]])
    df_y_cutoff.rename(columns={"y": "y_last", "ds": "cutoff"}, inplace=True)

    # Merge y of cutoff
    tmp_df = tmp_df.merge(df_y_cutoff, on=["unique_id", "cutoff"], how="left")

    # Reconstruct odometer levels
    tmp_df = _reconstruct_full_forecast(tmp_df)

    return tmp_df
