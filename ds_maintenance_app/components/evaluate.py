import numpy as np
import pandas as pd


def calculate_mae(
    df_backtest_results: pd.DataFrame,
    target_column: str = "y",
    pred_column: str = "yhat",
) -> float:
    """
    Calculate the Mean Absolute Error (MAE) across multiple time series.

    The MAE is computed globally across all observations:
        MAE = sum(|y - y_pred|) / N

    __PARAMETERS__
    df_backtest_results : pd.DataFrame
        DataFrame containing the forecast results.
    target_column : str, default="y"
        Column representing the actual values.
    pred_column : str, default="yhat"
        Column representing the predicted values.
    id_column : str, default="unique_id"
        Column identifying unique time series (kept for compatibility).

    __RETURNS__
    float
        The Mean Absolute Error (MAE) across all observations.
    """

    total_abs_error = (
        (df_backtest_results[target_column] - df_backtest_results[pred_column])
        .abs()
        .sum()
    )
    total_points = len(df_backtest_results)

    if total_points == 0:
        return float("nan")

    return float(total_abs_error / total_points)


def calculate_mean_absolute_scaled_error_h(
    df_backtest_results: pd.DataFrame,
    df_stats: pd.DataFrame,
    target_column: str = "y",
    pred_column: str = "yhat",
    id_column: str = "unique_id",
    step: int = 1,
) -> float:
    """
    Calculate a volume-weighted Mean Absolute Scaled Error (MASE)
    across multiple time series.

    The function calculates:
    - The Mean Absolute Error (MAE) between `y_test` and `y_pred`
    - Scales it by the MAE of a naive forecast computed on `y_train`
      using a lag of size `step`

    __PARAMETERS__
    df_backtest_results : pd.DataFrame
        DataFrame containing the forecast results, including columns for
        the target (`target_column`) and predictions (`pred_column`).
    df_stats : pd.DataFrame
        DataFrame containing historical training data for each series.
        Used to calculate the scaling factor for MASE.
    target_column : str, default="y"
        Column name in `df_backtest_results` representing the actual values.
    pred_column : str, default="yhat"
        Column name in `df_backtest_results` representing the predicted values.
    id_column : str, default="unique_id"
        Column name identifying unique time series.
    step : int, default=1
        Lag step used in the naive forecast for scaling.

    __RETURNS__
    float
        The Mean Absolute Scaled Error (MASE).
        Returns np.inf if the scaling factor is zero.
    """

    # Pre-group series and convert each to a numpy arrays for faster computation
    y_test_dict = (
        df_backtest_results.groupby(id_column)[target_column].apply(np.array).to_dict()
    )
    y_pred_dict = (
        df_backtest_results.groupby(id_column)[pred_column].apply(np.array).to_dict()
    )
    y_train_dict = df_stats.groupby(id_column)[target_column].apply(np.array).to_dict()

    total_abs_error = 0.0  # Numerator: MAE on test
    total_naive_error = 0.0  # Denominator: MAE of naive forecast on training

    for unique_id in y_test_dict:

        # Extract series series values from dicts
        y_test = y_test_dict[unique_id]
        y_pred = y_pred_dict[unique_id]
        y_train = y_train_dict[unique_id]

        # Skip series with too-short or flat training data
        if len(y_train) <= step or np.all(y_train == y_train[0]):
            continue

        # Add series MAE to numerator
        total_abs_error += np.sum(np.abs(y_test - y_pred))

        # Add MAE of naive forecast to denominator
        naive_errors = np.abs(y_train[step:] - y_train[:-step])
        total_naive_error += np.sum(naive_errors)

    if total_naive_error == 0:
        return float(np.inf)

    return float(total_abs_error / total_naive_error)


def calculate_wmape(
    df_backtest_results: pd.DataFrame,
    target_column: str = "y",
    pred_column: str = "yhat",
) -> float:
    """
    Calculate the global Weighted Mean Absolute Percentage Error (wMAPE)
    across multiple time series.

    Global wMAPE is defined as:
        wMAPE = sum(|y_test - y_pred|) / sum(|y_test|)

    This weights all observations by their magnitude rather than averaging
    per-series errors.

    __PARAMETERS__
    df_backtest_results : pd.DataFrame
        DataFrame containing the forecast results.
    target_column : str, default="y"
        Column representing the actual values.
    pred_column : str, default="yhat"
        Column representing the predicted values.

    __RETURNS__
    float
        The global wMAPE across all observations.
    """

    total_abs_error = (
        (df_backtest_results[target_column] - df_backtest_results[pred_column])
        .abs()
        .sum()
    )
    total_magnitude = df_backtest_results[target_column].abs().sum()

    if total_magnitude == 0:
        return float("nan")

    return float(total_abs_error / total_magnitude)


def calculate_average_bias(
    df_backtest_results: pd.DataFrame,
    target_column: str = "y",
    pred_column: str = "yhat",
    id_column: str = "unique_id",
) -> float:
    """
    Calculate the average normalized bias across multiple time series.

    The bias for a single series is defined as:
        bias = sum(y_pred - y_test) / sum(|y_test|)

    __PARAMETERS__
    df_backtest_results : pd.DataFrame
        DataFrame containing the forecast results, including columns for
        the target (`target_column`) and predictions (`pred_column`).
    target_column : str, default="y"
        Column name in `df_backtest_results` representing the actual values.
    pred_column : str, default="yhat"
        Column name in `df_backtest_results` representing the predicted values.
    id_column : str, default="unique_id"
        Column name identifying unique time series.

    __RETURNS__
    float
        The mean normalized bias across all valid time series.
    """

    # Convert each series into numpy arrays for faster computation
    y_test_dict = (
        df_backtest_results.groupby(id_column)[target_column].apply(np.array).to_dict()
    )
    y_pred_dict = (
        df_backtest_results.groupby(id_column)[pred_column].apply(np.array).to_dict()
    )

    bias_results = []

    for unique_id in y_test_dict:
        y_test = y_test_dict[unique_id]
        y_pred = y_pred_dict[unique_id]

        denom = np.sum(np.abs(y_test))

        # Skip if denom == 0
        if denom == 0:
            continue

        # Compute bias
        bias = np.sum(y_pred - y_test) / denom

        bias_results.append({"unique_id": unique_id, "bias": bias})

    # Convert final results to DataFrame
    df_bias = pd.DataFrame(bias_results)

    # Calculate mean of all bias values
    mean_bias = df_bias["bias"].mean()

    return float(mean_bias)


def calculate_global_bias(
    df_backtest_results: pd.DataFrame,
    target_column: str = "y",
    pred_column: str = "yhat",
) -> float:
    """
    Calculate the global normalized bias across the entire dataset.

    Global bias is defined as:
        bias = sum(y_pred - y_actual) / sum(|y_actual|)

    __PARAMETERS__
    df_backtest_results : pd.DataFrame
        DataFrame containing the forecast results, including columns for
        the target (`target_column`) and predictions (`pred_column`).
    target_column : str, default="y"
        Column name in `df_backtest_results` representing the actual values.
    pred_column : str, default="yhat"
        Column name in `df_backtest_results` representing the predicted values.

    __RETURNS__
    float
        The global normalized bias.
    """
    y_actual = df_backtest_results[target_column].to_numpy()
    y_pred = df_backtest_results[pred_column].to_numpy()

    denom = np.sum(np.abs(y_actual))

    if denom == 0:
        return 0.0

    bias = np.sum(y_pred - y_actual) / denom

    return float(bias)


def calculate_evaluation_metrics(df_backtest_results, df_stats):
    """
    Calculate multiple evaluation metrics (MASE, wMAPE, and bias) for forecast results.

    __PARAMETERS__
    df_backtest_results : pd.DataFrame
        DataFrame containing forecast results, including columns for actual values
        (`y`) and predicted values (`yhat`), and a unique series identifier (`unique_id`).
    df_stats : pd.DataFrame
        DataFrame containing historical training data for each series, used for MASE scaling.

    __RETURNS__
    pd.DataFrame containing a single value for:
        - MASE: Mean Absolute Scaled Error
        - wMAPE: Weighted Mean Absolute Percentage Error
        - bias: Normalized bias
        - backtest_date: The current date
    """

    # Calculate MASE
    mae = calculate_mae(df_backtest_results)

    # Calculate MASE
    mase = calculate_mean_absolute_scaled_error_h(df_backtest_results, df_stats)

    # Calculate wMAPE
    wmape = calculate_wmape(df_backtest_results)

    # Calculate average bias
    average_bias = calculate_average_bias(df_backtest_results)

    # Calculate global bias
    global_bias = calculate_global_bias(df_backtest_results)

    return pd.DataFrame(
        {
            "MAE": [mae],
            "MASE": [mase],
            "wMAPE": [wmape],
            "average_bias": [average_bias],
            "global_bias": [global_bias],
        }
    )
