from typing import Dict, List
import pandas as pd
from copy import deepcopy
import xgboost as xgb
from mlforecast.lag_transforms import RollingMean, RollingStd, ExpandingMean

from ds_package_forecast.ml_models import MLModels
from ds_package_forecast.utils import exclude_too_short_items


def prepare_features_for_forecaster(
    df_features: pd.DataFrame,
    target: str,
    id_column: str,
    date_column: str,
    static_features: List[str],
    dynamic_features: List[str],
) -> pd.DataFrame:
    """
    Selects and formats a feature DataFrame for use with MLForecast.

    The function:
    - Selects only the required feature columns
    - Renames columns to MLForecast-compatible names:
        id_column -> "unique_id"
        date_column -> "ds"
        target -> "y"
    - Casts static feature columns to categorical dtype

    __PARAMETERS__
    df_features: pd.DataFrame
        Input DataFrame containing time-series features and target
    target: str
        Name of the target column
    id_column: str
        Name of the column containing the series identifiers
    date_column: str
        Name of the datetime column
    static_features: List[str]
        List of static feature column names
    dynamic_features: List[str]
        List of dynamic (time-varying) feature column names

    __RETURNS__
    pd.DataFrame
        A formatted DataFrame compatible with MLForecast, containing:
        - unique_id
        - ds
        - y
        - static features
        - dynamic features
    """

    # Define features and target for forecast
    all_features = (
        [id_column, date_column] + static_features + dynamic_features + [target]
    )

    # Format table for MLForecast
    df_stats = deepcopy(df_features[all_features])
    df_stats.rename(
        columns={id_column: "unique_id", date_column: "ds", target: "y"}, inplace=True
    )
    df_stats["ds"] = pd.to_datetime(df_stats["ds"])

    # Fix type of static features
    for feature in static_features:
        df_stats[feature] = df_stats[feature].astype("category")

    # Remove null values from df_stats_odometer
    df_stats.dropna(inplace=True)

    # Sort by id and ds
    df_stats.sort_values(by=["unique_id", "ds"], inplace=True)

    # Reset index
    df_stats.reset_index(drop=True, inplace=True)

    return df_stats


def prepare_future_table(
    df_future: pd.DataFrame,
    id_column: str,
    date_column: str,
    dynamic_features: List[str],
) -> pd.DataFrame:
    """
    Formats a future feature DataFrame for use with MLForecast
    prediction or inference.

    The function:
    - Renames identifier and date columns to MLForecast-compatible names:
        id_column -> "unique_id"
        date_column -> "ds"
    - Selects only the required columns (ID, date, and dynamic features)
    - Returns a table ready to be passed as `X_df` for future forecasting

    __PARAMETERS__
    df_future: pd.DataFrame
        Input DataFrame containing future timestamps and dynamic features
    id_column: str
        Name of the column containing the series identifiers
    date_column: str
        Name of the datetime column
    dynamic_features: List[str]
        List of dynamic (time-varying) feature column names to include

    __RETURNS__
    pd.DataFrame
        A formatted DataFrame compatible with MLForecast future input,
        containing:
        - unique_id
        - ds
        - dynamic features
    """

    X_future = deepcopy(df_future)

    # Prepare X_future
    X_future.rename(columns={id_column: "unique_id", date_column: "ds"}, inplace=True)
    X_future["ds"] = pd.to_datetime(X_future["ds"])

    X_future = X_future[["unique_id", "ds"] + dynamic_features]

    return X_future


def first_order_difference_target(df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Converts an absolute target time series into first differences per entity.

    This transformation is commonly used when modeling trends and seasonality using a tree-based model.

    __PARAMETERS__
    df_stats: pd.DataFrame
        Input DataFrame formatted for MLForecast, containing:
        - unique_id
        - ds
        - y
        - optional feature columns

    __RETURNS__
    pd.DataFrame
        DataFrame with the same structure as input but with the target column
        replaced by first differences.
    """

    df_stats_delta = deepcopy(df_stats)

    # generate y delta values
    df_stats_delta["y_delta"] = (
        df_stats_delta.groupby("unique_id")["y"].diff().fillna(0.0)
    )

    # Replace target
    df_stats_delta = df_stats_delta.drop(columns="y").rename(columns={"y_delta": "y"})

    return df_stats_delta


def build_forecasters(
    df_stats_delta: pd.DataFrame,
    horizon: int,
    n_windows_list: list[int | None],
    model_hyperparameters: dict,
    lags: list[int],
    date_features: list[str],
    static_features: list[str],
) -> dict[int | None, MLModels]:
    """
    Builds multiple MLModels forecasters using different rolling window configurations.
    The goal is to split the data into groups depending on their size so that CI calculations
    can be applied with a larger number of windows to groups with more data.

    The function:
    - Iterates over a list of window sizes used for calculating confidence intervals
    - Filters out time series that are too short for the given
      horizon and window configuration
    - Instantiates an XGBoost-based MLModels forecaster for each window size
    - Returns a dictionary mapping window size to configured forecaster

    __PARAMETERS__
    df_stats_delta: pd.DataFrame
        Input DataFrame containing first-differenced target values,
        formatted for MLForecast.
    horizon: int
        Forecast horizon (number of future time steps).
    n_windows_list: list[int | None]
        List of rolling window counts for cross-validation.
        Use None to indicate training on the full dataset.
    model_hyperparameters: dict
        Dictionary of hyperparameters passed to XGBRegressor.
    lags: list[int]
        List of integer lag values to include as autoregressive features.
    date_features: list[str]
        List of date-based feature names to extract from the timestamp column.
    static_features: list[str]
        List of column names representing static (entity-level) features.

    __RETURNS__
    dict[int | None, MLModels]
        Dictionary mapping each window size to a configured MLModels forecaster.
    """

    # Step size for CI calculation
    step_size = horizon

    # define lag transforms
    lag_transforms = {
        1: [
            ExpandingMean(),
            RollingMean(window_size=7),
            RollingStd(window_size=7),
        ]
    }

    forecasters = {}
    for n_windows in n_windows_list:

        X = deepcopy(df_stats_delta)

        # Filter usable series if n_windows is not zero
        if n_windows > 1:
            # Filter usable series with window size
            X = exclude_too_short_items(
                X, h=horizon, n_windows=n_windows, step_size=step_size
            )

        # Define forecaster
        forecaster = MLModels(
            df=X,
            model={"xgb": xgb.XGBRegressor(**model_hyperparameters)},
            freq="D",  # Daily frequency
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            static_features=static_features,
        )
        forecasters[n_windows] = forecaster

    return forecasters


def execute_forecasters(
    forecasters: Dict[int | None, object],
    n_windows_list: List[int | None],
    X_future: pd.DataFrame,
    horizon: int,
    level: List[int],
) -> pd.DataFrame:
    """
    Executes multiple forecasters across different rolling window configurations
    and combines their results into a single DataFrame. Applies CI calculations
    to the series that have enough data.

    The function:
    - Iterates over a list of rolling window sizes
    - Runs forecasts for each configured forecaster
    - Prevents duplicate forecasting of the same unique_id across window sizes
    - Disables confidence intervals when insufficient windows are available
    - Tracks which window configuration produced each forecast
    - Returns a concatenated DataFrame containing all forecast outputs

    __PARAMETERS__
    forecasters: dict[int | None, object]
        Dictionary mapping window size to a configured forecaster object.
        Each forecaster must implement a `.forecast()` method compatible
        with MLForecast-style arguments.
    n_windows_list: list[int | None]
        List of rolling window counts to execute.
        Typically ordered from smallest to largest window size.
    X_future: pd.DataFrame
        Future exogenous feature DataFrame containing at least:
        - unique_id
        - timestamp column
        - any required exogenous features
    horizon: int
        Forecast horizon (number of future time steps).
    level: List[int]
        Confidence interval levels (e.g., [80, 95]).
        Automatically disabled when n_windows < 2.

    __RETURNS__
    pd.DataFrame
        Combined forecast results for all window configurations,
        including a column `n_windows` indicating which configuration
        produced each forecast.
    """

    # Record forecasted vins
    forecasted_vins = []

    # Record results from each forecaster
    forecast_results = []

    # forecast each model
    for n_windows in n_windows_list:
        print("forecasting with n_windows =", n_windows)

        # set level to none if too few windows for CI calculation
        if n_windows < 2:
            level = None
        else:
            level = level

        # forecast
        df_forecast_results = forecasters[n_windows].forecast(
            h=horizon, X_future=X_future, level=level, n_windows=n_windows
        )

        # Drop vins already forecasted
        df_forecast_results = df_forecast_results[
            ~df_forecast_results["unique_id"].isin(forecasted_vins)
        ]
        print("newly forecasted vins:", len(df_forecast_results.unique_id.unique()))

        # Record vins used in forecast
        forecasted_vins = forecasted_vins + list(df_forecast_results.unique_id.unique())

        # record n_windows of series
        df_forecast_results["n_windows"] = n_windows

        forecast_results.append(df_forecast_results)

        df_forecast_results = pd.concat(forecast_results)

    return df_forecast_results


def backtest(
    df_stats_delta: pd.DataFrame,
    horizon: int,
    n_windows: int,
    model_hyperparameters: dict,
    lags: list[int],
    date_features: list[str],
    static_features: list[str],
) -> pd.DataFrame:
    """
    Builds and executes a backtest using an XGBoost-based MLModels forecaster.

    The function:
    - Defines lag-based transformations for autoregressive features
    - Instantiates an MLModels forecaster with the specified features and hyperparameters
    - Performs backtesting with the given horizon and rolling window configuration
    - Returns the backtest results for all series in a DataFrame

    __PARAMETERS__
    df_stats_delta: pd.DataFrame
        Input DataFrame containing first-differenced target values ("y") and
        formatted for MLForecast with columns:
        - unique_id
        - ds
        - y
        - optional static and dynamic features
    horizon: int
        Forecast horizon (number of future time steps to backtest)
    n_windows: int
        Number of rolling windows to use for backtesting
    model_hyperparameters: dict
        Dictionary of hyperparameters passed to XGBRegressor
    lags: list[int]
        List of integer lag values to include as autoregressive features
    date_features: list[str]
        List of date-based feature names to extract from the timestamp column
    static_features: list[str]
        List of column names representing static (entity-level) features

    __RETURNS__
    pd.DataFrame
        Backtest results containing predictions for each series and timestamp,
        formatted similarly to MLForecast output.
    """

    # Step size for backtest
    step_size = round(horizon / 2)

    # define lag transforms
    lag_transforms = {
        1: [
            ExpandingMean(),
            RollingMean(window_size=7),
            RollingStd(window_size=7),
        ]
    }

    X = exclude_too_short_items(
        df_stats_delta, h=horizon, n_windows=n_windows, step_size=step_size
    )

    # Define forecaster
    forecaster = MLModels(
        df=X,
        model={"xgb": xgb.XGBRegressor(**model_hyperparameters)},
        freq="D",  # Daily frequency
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=date_features,
        static_features=static_features,
    )

    # Backtest
    df_backtest_results = forecaster.backtest(
        h=horizon, n_windows=n_windows, step_size=step_size
    )

    return df_backtest_results
