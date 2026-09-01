from copy import deepcopy
import pandas as pd
from typing import Tuple


def predict_oil_change_date_from_forecast(
    df_oil_percent_forecast_results_post: pd.DataFrame,
    forecast_column: str = "low",
    id_column: str = "unique_id",
    date_column: str = "ds",
):
    """
    Predict the first date when the forecasted oil percentage reaches zero or below.

    __PARAMETERS__
    df_oil_percent_forecast_results_post: pd.DataFrame
        DataFrame containing forecasted oil percentage values per vehicle
    forecast_column: str
        Column name containing the forecasted oil percentage values
    id_column: str
        Column name identifying each unique vehicle
    date_column: str
        Column name containing the forecast dates

    __RETURNS__
    pd.DataFrame
        A DataFrame containing one row per vehicle with the first date
        where the forecasted oil percentage is less than or equal to zero.
        The result is sorted by the predicted oil change date.
    """

    tmp_df = deepcopy(df_oil_percent_forecast_results_post)

    # get cars forecasted to reach oil change percent zero
    tmp_df = (
        tmp_df[tmp_df[forecast_column] <= 0]
        .sort_values(by=[id_column, date_column])
        .groupby("unique_id", as_index=False)[date_column]
        .first()
        .sort_values(by=date_column)
    )

    return tmp_df


def calculate_latest_service_distances(
    df_features: pd.DataFrame,
    service_distance_column: str,
    id_column: str = "vin",
    date_column: str = "updated_date",
) -> pd.DataFrame:
    """
    Calculate the adjusted (current) distance-to-service by subtracting the
    distance travelled since the last recorded distance-to-service measurement
    from the TCU.

    The travelled distance is inferred from the change in gmx_odometer between
    the last recorded service distance date and the most recent available date
    per vehicle.

    __PARAMETERS__
    df_features: pd.DataFrame
        DataFrame containing vehicle features including service distance
        and odometer readings
    service_distance_column: str
        Column name containing the recorded distance-to-service values
    id_column: str
        Column name identifying each unique vehicle
    date_column: str
        Column name containing the timestamp of the feature record

    __RETURNS__
    pd.DataFrame
        A DataFrame with one row per vehicle containing:
        - unique_id
        - The adjusted distance-to-service value (updated by subtracting
          the distance travelled since the last recorded value)
    """

    tmp_df = deepcopy(df_features)

    # Rename id column to match MLForecast output
    tmp_df.rename(columns={id_column: "unique_id"}, inplace=True)

    # Sort features
    tmp_df.sort_values(by=["unique_id", date_column], inplace=True)

    # Get last measurements of distance to service for each car
    df_tcu_measurement_last = tmp_df[
        ["unique_id", date_column, service_distance_column, "gmx_odometer"]
    ].copy()
    df_tcu_measurement_last.dropna(inplace=True)
    df_tcu_measurement_last = df_tcu_measurement_last.groupby(
        "unique_id", as_index=False
    ).last()
    df_tcu_measurement_last.rename(
        columns={
            service_distance_column: f"{service_distance_column}_last_recorded",
            date_column: f"last_recorded_{service_distance_column}_date",
            "gmx_odometer": "gmx_odometer_on_last_recorded_date",
        },
        inplace=True,
    )

    # Get last gmx_odometer values for each car
    df_current_odometer_values = tmp_df[
        ["unique_id", date_column, "gmx_odometer"]
    ].copy()
    df_current_odometer_values = df_current_odometer_values.groupby(
        "unique_id", as_index=False
    ).last()
    df_current_odometer_values.rename(
        columns={date_column: "current_date", "gmx_odometer": "gmx_odometer_current"},
        inplace=True,
    )

    # Merge last odometer values
    df_tcu_measurement_last = df_tcu_measurement_last.merge(
        df_current_odometer_values, on="unique_id", how="left"
    )

    # Calculate change in odometer and subtract from last distance to service to estimate current distance to service
    df_tcu_measurement_last["gmx_odometer_difference"] = (
        df_tcu_measurement_last["gmx_odometer_current"]
        - df_tcu_measurement_last["gmx_odometer_on_last_recorded_date"]
    )
    df_tcu_measurement_last[f"{service_distance_column}_last_recorded"] = (
        df_tcu_measurement_last[f"{service_distance_column}_last_recorded"]
        - df_tcu_measurement_last["gmx_odometer_difference"]
    )

    # Cleanup columns
    df_tcu_measurement_last.drop(
        columns=[
            f"last_recorded_{service_distance_column}_date",
            "gmx_odometer_on_last_recorded_date",
            "current_date",
            "gmx_odometer_current",
            "gmx_odometer_difference",
        ],
        inplace=True,
    )

    return df_tcu_measurement_last


def estimate_future_service_distance(
    df_forecast_results_odometer_post: pd.DataFrame,
    df_service_distance_latest: pd.DataFrame,
    service_distance_column: str,
    target_delta: str = "high_delta",
) -> pd.DataFrame:
    """
    Estimate the future distance-to-service by applying forecasted odometer
    deltas to the last recorded service distance.

    The function cumulatively sums the forecasted odometer deltas per vehicle
    and subtracts them from the latest recorded distance-to-service value to
    estimate the future remaining service distance.

    __PARAMETERS__
    df_forecast_results_odometer_post: pd.DataFrame
        DataFrame containing forecasted odometer deltas per vehicle and date
    df_service_distance_latest: pd.DataFrame
        DataFrame containing the latest recorded service distance per vehicle.
        Must include 'unique_id' and '{service_distance_column}_last_recorded'
    service_distance_column: str
        Base name of the service distance column (without suffix)
    target_delta: str
        Column name containing the forecasted odometer delta values

    __RETURNS__
    pd.DataFrame
        A DataFrame containing the original forecast rows along with an
        additional column '{service_distance_column}_estimate' representing
        the estimated future distance-to-service.
    """

    tmp_df = df_forecast_results_odometer_post.copy()

    # merge last service distance recorded
    tmp_df = tmp_df.merge(df_service_distance_latest, on="unique_id", how="left")
    tmp_df.dropna(inplace=True)

    # Estimate decrease in distance to service using forecasted odometer delta
    tmp_df[f"{service_distance_column}_estimate"] = (
        tmp_df[f"{service_distance_column}_last_recorded"]
        - tmp_df.groupby("unique_id")[target_delta].cumsum()
    )

    return tmp_df


def get_upcoming_service_dates(
    df: pd.DataFrame,
    service_distance_forecast_column: str,
    id_column="unique_id",
    date_column="ds",
) -> pd.DataFrame:
    """
    Get vehicles estimated to require service within the forecast horizon.

    The function filters the DataFrame to rows where the forecasted service
    distance is less than or equal to zero, then selects the earliest date
    per vehicle when service is expected. The result is sorted by date.

    __PARAMETERS__
    df: pd.DataFrame
        DataFrame containing vehicle forecasts, must include:
        - 'unique_id' column identifying each vehicle
        - 'ds' column containing forecast dates
        - The column specified by `service_distance_forecast_column`
    service_distance_forecast_column: str
        Column name containing the forecasted remaining distance-to-service

    __RETURNS__
    pd.DataFrame
        A DataFrame with one row per vehicle containing:
        - 'unique_id'
        - 'ds': the earliest date when the vehicle is estimated to require service
        Sorted by 'ds' ascending.
    """
    tmp_df = deepcopy(df)

    tmp_df = (
        tmp_df[tmp_df[service_distance_forecast_column] <= 0]
        .sort_values(by=[id_column, date_column])
        .groupby(id_column, as_index=False)[date_column]
        .first()
        .sort_values(by=date_column)
    )

    return tmp_df


def concat_oil_service_predictions(
    df_upcoming_oil_percent_service_dates: pd.DataFrame,
    df_upcoming_oil_distance_service_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenate upcoming oil service prediction DataFrames into a single DataFrame.

    __PARAMETERS__
    df_upcoming_oil_percent_service_dates: pd.DataFrame
        DataFrame containing upcoming oil service dates predicted based on
        percentage of oil remaining.
    df_upcoming_oil_distance_service_dates: pd.DataFrame
        DataFrame containing upcoming oil service dates predicted based on
        distance traveled since last oil change.

    __RETURNS__
    pd.DataFrame
    """

    # Concat tables and reset index
    df_upcoming_oil_service_dates = pd.concat(
        [df_upcoming_oil_percent_service_dates, df_upcoming_oil_distance_service_dates],
        axis=0,
    )
    df_upcoming_oil_service_dates.reset_index(drop=True, inplace=True)

    return df_upcoming_oil_service_dates


def get_last_serviceable_date(
    df_upcoming_service_dates: pd.DataFrame, X_future: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the last serviceable date for each vehicle before its next scheduled service.

    The function filters future forecast rows where the vehicle is not booked and
    the date is before the predicted service-needed date, then selects the latest
    such date per vehicle. The result is merged back to the original upcoming
    services DataFrame.

    __PARAMETERS__
    df_upcoming_service_dates: pd.DataFrame
        DataFrame containing upcoming service predictions. Must include:
        - 'unique_id' column identifying each vehicle
        - 'ds_service_needed' column specifying the predicted service date
    X_future: pd.DataFrame
        DataFrame of future forecast data. Must include:
        - 'unique_id' column
        - 'ds' column with forecast dates
        - 'is_booked' column indicating whether the vehicle is booked (0 = not booked)

    __RETURNS__
    pd.DataFrame
        The df_last_service_opportunity_dates DataFrame with an additional column:
        - 'last_serviceable_date': the latest date before service is needed and
          the vehicle is not booked
    """

    # Keep only relevant columns from df_upcoming_service_dates
    tmp_df = deepcopy(X_future[["unique_id", "ds", "is_booked"]])

    # Inner join predicted service date into future features
    tmp_df = tmp_df.merge(
        df_upcoming_service_dates,
        on="unique_id",
        how="inner",
        suffixes=("", "_service_needed"),
    )

    # Keep only rows before the service needed date AND where is_booked == 0
    tmp_df = tmp_df[
        (tmp_df["ds"] < tmp_df["ds_service_needed"]) & (tmp_df["is_booked"] == 0)
    ]

    # For each car, get the last date
    tmp_df = (
        tmp_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id", as_index=False)["ds"]
        .last()
        .rename(columns={"ds": "last_serviceable_date"})
    )

    # Merge back to your result
    df_last_service_opportunity_dates = df_upcoming_service_dates.merge(
        tmp_df, on="unique_id", how="left"
    )

    return df_last_service_opportunity_dates


def prepare_output_tables(
    df_last_vehicle_service_opportunity_dates: pd.DataFrame,
    df_last_oil_service_opportunity_dates: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare output tables for upcoming vehicle and oil service dates.

    This function renames columns in the input DataFrames to standardized output
    names for reporting or downstream use.

    __PARAMETERS__
    df_last_vehicle_service_opportunity_dates: pd.DataFrame
        DataFrame containing the last predicted vehicle service dates.
        Must include columns:
        - 'unique_id': identifier for each vehicle
        - 'ds': predicted service date
    df_last_oil_service_opportunity_dates: pd.DataFrame
        DataFrame containing the last predicted oil service dates.
        Must include columns:
        - 'unique_id': identifier for each vehicle
        - 'ds': predicted oil service date

    __RETURNS__
    Tuple[pd.DataFrame, pd.DataFrame]
        - First DataFrame: vehicle service output table with columns:
        - Second DataFrame: oil service output table with columns:
    """

    # Rename columns for upcoming vehicle service dates
    df_vehicle_service_output = deepcopy(df_last_vehicle_service_opportunity_dates)
    df_vehicle_service_output.rename(
        columns={"unique_id": "vin", "ds": "forecasted_vehicle_service_date"},
        inplace=True,
    )

    # Rename columns for upcoming oil service dates
    df_oil_service_output = deepcopy(df_last_oil_service_opportunity_dates)
    df_oil_service_output.rename(
        columns={"unique_id": "vin", "ds": "forecasted_oil_service_date"}, inplace=True
    )

    return df_vehicle_service_output, df_oil_service_output
