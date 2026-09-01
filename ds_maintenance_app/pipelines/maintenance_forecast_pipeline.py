from ds_maintenance_app.pipelines.base import BasePipeline
from ds_maintenance_app.components.data import (
    load_bigquery_features,
    save_table_to_bigquery,
    append_table_to_bigquery,
)
from ds_maintenance_app.utils.utils import check_valid_measurements
from ds_maintenance_app.components.preprocess import (
    nullify_zeros,
    nullify_if_other_not_null,
    convert_booleans,
    split_future_data,
    nullify_before_long_gaps,
    hampel_filter_forecasting_columns,
    nullify_decreases,
    nullify_all_values_before_decreases,
    nullify_before_unrealistic_changes,
    remove_resets_from_service_columns,
    interpolate_target_columns_when_booked,
    forward_fill_missing,
    drop_cars_without_measurements,
)
from ds_maintenance_app.components.forecast import (
    prepare_features_for_forecaster,
    prepare_future_table,
    first_order_difference_target,
    backtest,
    build_forecasters,
    execute_forecasters,
)
from ds_maintenance_app.components.postprocess import (
    postprocess_forecast,
    postprocess_backtest,
)
from ds_maintenance_app.components.evaluate import calculate_evaluation_metrics

from ds_maintenance_app.components.output import (
    predict_oil_change_date_from_forecast,
    calculate_latest_service_distances,
    estimate_future_service_distance,
    get_upcoming_service_dates,
    concat_oil_service_predictions,
    prepare_output_tables,
    get_last_serviceable_date,
)
from ds_maintenance_app.components.monitoring import (
    detect_resets_in_service_columns,
    detect_zero_crossings_in_service_columns,
)


class MaintenanceForecastPipeline(BasePipeline):

    def __init__(
        self,
        env: str = "dev",
        pipeline_name="Maintenance Forecast Pipeline",
    ):
        """
        A pipeline class for preprocessing, forecasting, and evaluating maintenance targets.

        This pipeline performs the following steps:

        Step 1 Load the data from BigQuery: {self.database}.{self.features_table}
            - Step 1.1 Load the data features: {self.database}.{self.features_table}

        Step 2 Prepare training data
            - Step 2.1 Sort values before applying preprocessing
            - Step 2.2 Convert boolean column values to 1 and 0
            - Step 2.3 Split future data
            - Step 2.4 Null all zero values
            - Step 2.5 Null values in gmx_service_oil_interval_distance when gmx_service_oil_interval_percent is measured
            - Step 2.6 Null all values in each series before large gaps of missing values
            - Step 2.7 Apply Hampel filter to each series to be used in forecaster
            - Step 2.8 Save copy of gmx_service_oil_interval_percent at this step for use in model monitoring ground truth labelling
            - Step 2.9 Apply filter to null all odometer values where a decrease is found to remove invalid changes from each series
            - Step 2.10 Apply filter to null all odometer values before where a decrease is found. This more aggressive filter is to clean any series that the previous filter could not clean fully
            - Step 2.11 Apply nullify_before_unrealistic_changes filter to remove changes in gmx_service_oil_interval_percent where decreases are too large or increases are too small to be resets due to service
            - Step 2.12 Apply nullify_before_unrealistic_changes filter to remove changes in gmx_odometer where increases are too large
            - Step 2.13 Process gmx_service_oil_interval_percent to remove value resets due to services
            - Step 2.14 Interpolate target values where is_booked = 1
            - Step 2.15 Forward fill uninterpolated missing values
            - Step 2.16 Drop cars that do not have any values in the service columns required for predictions
            - Step 2.17 Check number of valid vins across each TCU measurement
            - Step 2.18 Save preprocessed features table to BigQuery
            - Step 2.19 Save preprocessed future features table to BigQuery
            - Step 2.20 Save counts of valid cars per TCU measurement to BigQuery

        Step 3 Backtest to generate evaluation metrics on current data
            - Step 3.1 Prepare training data for odometer forecast
            - Step 3.2 Prepare training data for oil percent forecast
            - Step 3.3 Prepare future data
            - Step 3.4 Apply first order differencing to the odometer training data
            - Step 3.5 Apply first order differencing to the oil percent training data
            - Step 3.6 Backtest gmx_odometer to calculate evaluation metrics for forecast
            - Step 3.7 Backtest gmx_service_oil_interval_percent to calculate evaluation metrics for forecast
            - Step 3.8 Postprocess backtest results of gmx_odometer
            - Step 3.9 Postprocess backtest results of gmx_service_oil_interval_percent
            - Step 3.10 Calculate evaluation metrics for backtested gmx_odometer results
            - Step 3.11 Calculate evaluation metrics for backtested gmx_service_oil_interval_percent results
            - Step 3.12 Save backtest forecast metrics to BigQuery

        Step 4 Forecast future TCU values
            - Step 4.1 Build odometer forecasters
            - Step 4.2 Build oil percent forecasters
            - Step 4.3 Execute odometer forecasters
            - Step 4.4 Execute oil percent forecasters
            - Step 4.5 Apply postprocessing to odometer forecast
            - Step 4.6 Apply postprocessing to oil percent forecast
            - Step 4.7 Save postprocessed forecasts

        Step 5 Calculate outputs from forecast
            - Step 5.1 Calculate dates of next needed oil changes from oil percent forecast
            - Step 5.2 Calculate distance travelled since last gmx_service_vehicle_interval_distance value was recorded for each car
            - Step 5.3 Calculate distance travelled since last gmx_service_oil_interval_distance value was recorded for each car.
            - Step 5.4 Estimate future gmx_service_vehicle_interval_distance using gmx_odometer forecast
            - Step 5.5 Estimate future gmx_service_oil_interval_distance using gmx_odometer forecast
            - Step 5.6 Save forecasts for service distances calculated from gmx_odometer forecast
            - Step 5.7 Calculate upcoming vehicle services
            - Step 5.8 Calculate upcoming oil services
            - Step 5.9 Concat upcoming oil service predictions
            - Step 5.10 Get last serviceable date for vehicle service
            - Step 5.11 Get last serviceable date for oil service
            - Step 5.12 Prepare prediction output tables
            - Step 5.13 Save upcoming vehicle services predictions to BigQuery
            - Step 5.14 Save upcoming oil services predictions to BigQuery

        Step 6 Generate model monitoring outputs
            - Step 6.1 Detect zero crossings in the service columns
            - Step 6.2 Detect resets in the service columns
            - Step 6.3 Save zero crossings table to BigQuery
            - Step 6.4 Save resets table to BigQuery

        __PARAMETERS__
        env: str
            The environment in which the pipeline is running (e.g., 'dev', 'main').
        pipeline_name: str
            A descriptive name for the pipeline instance.
        """

        super().__init__(env, pipeline_name)
        self.env = env

        # Preprocess config constants
        self.database = self.config["data"]["database"]
        self.features_table = self.config["data"]["features_table"]
        self.measurement_columns = self.config["data"]["measurement_columns"]
        self.id_column = self.config["data"]["id_column"]
        self.substitution_target_column = self.config["preprocess"][
            "measurement_substitution"
        ]["target"]
        self.substitution_reference_column = self.config["preprocess"][
            "measurement_substitution"
        ]["reference"]
        self.date_column = self.config["data"]["date_column"]
        self.blocked_column = self.config["preprocess"]["blocked_column"]
        self.gap_cutoff_days = self.config["preprocess"]["gap_cutoff_days"]
        self.dynamic_features = self.config["maintenance_forecast"]["dynamic_features"]
        self.static_features = self.config["maintenance_forecast"]["static_features"]
        self.hampel_config_dict = self.config["preprocess"]["hampel_filter"]
        self.odometer_column = self.config["preprocess"]["odometer"]["column"]
        self.odometer_decrease_cutoff = self.config["preprocess"]["odometer"][
            "decrease_cutoff"
        ]
        self.nullify_unrealistic_changes_decaying_dict = self.config["preprocess"][
            "nullify_unrealistic_changes_decaying"
        ]
        self.nullify_unrealistic_changes_rising_dict = self.config["preprocess"][
            "nullify_unrealistic_changes_rising"
        ]
        self.remove_resets_columns = self.config["preprocess"]["remove_resets_columns"]
        self.target_columns = self.config["preprocess"]["target_columns"]
        self.booked_column = self.config["preprocess"]["booked_column"]
        self.service_columns = self.config["preprocess"]["service_columns"]
        self.preprocessed_output_training_table = self.config["data"][
            "preprocess_output"
        ]["features_table"]
        self.preprocessed_output_future_table = self.config["data"][
            "preprocess_output"
        ]["future_table"]
        self.metrics_output_table = self.config["data"]["metrics_table"]
        self.valid_cars = self.config["data"]["valid_cars_table"]

        # Forecasting config constants
        self.odometer_target = self.config["maintenance_forecast"]["odometer"]["target"]
        self.oil_percent_target = self.config["maintenance_forecast"]["oil_percent"][
            "target"
        ]
        self.xgb_hyperparameters = self.config["maintenance_forecast"][
            "xgb_hyperparameters"
        ]
        self.lags = self.config["maintenance_forecast"]["lags"]
        self.date_features = self.config["maintenance_forecast"]["date_features"]
        self.horizon = self.config["maintenance_forecast"]["horizon"]
        self.n_windows_list = self.config["maintenance_forecast"]["n_windows_list"]
        self.backtest_horizon = self.config["maintenance_forecast"]["backtest"][
            "horizon"
        ]
        self.backtest_n_windows = self.config["maintenance_forecast"]["backtest"][
            "n_windows"
        ]

        # Postprocess config constants
        self.postprocess_dict = self.config["postprocess"]

        # Output config constants
        self.vehicle_service_distance_column = self.config["output"][
            "vehicle_service_distance"
        ]
        self.oil_service_distance_column = self.config["output"]["oil_service_distance"]
        self.output_database = self.config["data"]["output_dataset"]

        # Monitoring config constants
        self.monitoring_database = self.config["data"]["monitoring_dataset"]
        self.zero_crossings_table = self.config["data"]["monitoring_output"][
            "zero_crossings_table"
        ]
        self.resets_table = self.config["data"]["monitoring_output"]["resets_table"]
        self.monitoring_columns = [
            f"{self.oil_percent_target}_ground_truth_state",
            self.vehicle_service_distance_column,
            self.oil_service_distance_column,
        ]

    def run_pipeline(self):

        # Step 1 load data and artifacts from GCP
        self.logger.info(
            f"Step 1 Load the data from BigQuery: {self.database}.{self.features_table}"
        )

        # Step 1.1 Load the data features from BigQuery
        self.logger.info(
            f"Step 1.1 Load the data features: {self.database}.{self.features_table}"
        )
        df_features = load_bigquery_features(self.database, self.features_table)

        # Step 2 Prepare training data
        self.logger.info("Step 2 Prepare training data")

        # Step 2.1 Sort values before applying preprocessing
        self.logger.info("Step 2.1 Sort values before applying preprocessing")
        df_features = df_features.sort_values(by=[self.id_column, self.date_column])

        # Step 2.2 Convert boolean Column values to 1 and 0
        self.logger.info("Step 2.2 Convert boolean column values to 1 and 0")
        df_features = convert_booleans(df_features)

        # Step 2.3 Split future data
        self.logger.info("Step 2.3 Split future data")
        df_features, df_future = split_future_data(
            df_features, self.id_column, self.date_column, self.dynamic_features
        )

        # Step 2.4 Null all zero values
        self.logger.info("Step 2.4 Null all zero values")
        df_features = nullify_zeros(df_features, self.measurement_columns)

        # Step 2.5 Null values in gmx_service_oil_interval_distance when gmx_service_oil_interval_percent is measured
        self.logger.info(
            "Step 2.5 Null values in gmx_service_oil_interval_distance when gmx_service_oil_interval_percent is measured"
        )
        df_features = nullify_if_other_not_null(
            df_features,
            self.id_column,
            self.substitution_target_column,
            self.substitution_reference_column,
        )

        # Step 2.6 Null all values in each series before large gaps of missing values
        self.logger.info(
            "Step 2.6 Null all values in each series before large gaps of missing values"
        )
        df_features = nullify_before_long_gaps(
            df_features,
            self.id_column,
            self.date_column,
            self.measurement_columns,
            self.blocked_column,
            self.gap_cutoff_days,
        )

        # Step 2.7 Apply Hampel filter to each series to be used in forecaster
        self.logger.info(
            "Step 2.7 Apply Hampel filter to each series to be used in forecaster"
        )
        df_features = hampel_filter_forecasting_columns(
            df_features, self.id_column, self.date_column, self.hampel_config_dict
        )

        # Step 2.8 Save copy of gmx_service_oil_interval_percent at this step for use in model monitoring ground truth labelling
        self.logger.info(
            "Step 2.8 Save copy of gmx_service_oil_interval_percent at this step for use in model monitoring ground truth labelling"
        )
        df_features[f"{self.oil_percent_target}_ground_truth_state"] = df_features[
            self.oil_percent_target
        ]

        # Step 2.9 Apply filter to null all odometer values where a decrease is found to remove invalid changes from each series
        self.logger.info(
            "Step 2.9 Apply filter to null all odometer values where a decrease is found to remove invalid changes from each series"
        )
        df_features = nullify_decreases(
            df_features,
            self.id_column,
            self.odometer_column,
            self.odometer_decrease_cutoff,
        )

        # Step 2.10 Apply nullify_decreased_measurements filter to apply more aggressive filtering to outliers missed by hampel filter
        self.logger.info(
            "Step 2.10 Apply filter to null all odometer values before where a decrease is found. This more aggressive filter is to clean any series that the previous filter could not clean fully"
        )
        df_features = nullify_all_values_before_decreases(
            df_features,
            self.id_column,
            self.odometer_column,
            self.odometer_decrease_cutoff,
        )

        # Step 2.11 Apply nullify_before_unrealistic_changes_decaying filter to remove changes in gmx_service_oil_interval_percent where decreases are too large or increases are too small to be resets due to service
        self.logger.info(
            "Step 2.11 Apply nullify_before_unrealistic_changes filter to remove changes in gmx_service_oil_interval_percent where decreases are too large or increases are too small to be resets due to service"
        )
        df_features = nullify_before_unrealistic_changes(
            df_features,
            self.id_column,
            self.nullify_unrealistic_changes_decaying_dict,
            mode="decaying",
        )

        # Step 2.12 Apply nullify_before_unrealistic_changes_rising filter to remove changes in gmx_odometer where increases are too large
        self.logger.info(
            "Step 2.12 Apply nullify_before_unrealistic_changes filter to remove changes in gmx_odometer where increases are too large"
        )
        df_features = nullify_before_unrealistic_changes(
            df_features,
            self.id_column,
            self.nullify_unrealistic_changes_rising_dict,
            mode="rising",
        )

        # Step 2.13 Process gmx_service_oil_interval_percent to remove value resets due to services
        self.logger.info(
            "Step 2.13 Process gmx_service_oil_interval_percent to remove value resets due to services"
        )
        df_features = remove_resets_from_service_columns(
            df_features, self.id_column, self.remove_resets_columns
        )

        # Step 2.14 Interpolate target values where is_booked = 1
        self.logger.info("Step 2.14 Interpolate target values where is_booked = 1")
        df_features = interpolate_target_columns_when_booked(
            df_features, self.id_column, self.booked_column, self.target_columns
        )

        # Step 2.15 Forward fill uninterpolated missing values
        self.logger.info("Step 2.15 Forward fill uninterpolated missing values")
        df_features = forward_fill_missing(
            df_features, self.id_column, self.target_columns
        )

        # Step 2.16 Drop cars that do not have any values in the service columns required for predictions
        self.logger.info(
            "Step 2.16 Drop cars that do not have any values in the service columns required for predictions"
        )
        df_features = drop_cars_without_measurements(
            df_features, self.id_column, self.date_column, self.service_columns
        )

        # Step 2.17 Check number of valid vins across each TCU measurement
        self.logger.info(
            "Step 2.17 Check number of valid vins across each TCU measurement"
        )
        df_valid_measurements = check_valid_measurements(
            df_features, self.measurement_columns, self.id_column
        )

        # Step 2.18 Save preprocessed features table to BigQuery
        self.logger.info("Step 2.18 Save preprocessed features table to BigQuery")
        save_table_to_bigquery(
            df_features,
            self.output_database,
            self.preprocessed_output_training_table,
            self.env,
        )

        # Step 2.19 Save preprocessed future features table to BigQuery
        self.logger.info(
            "Step 2.19 Save preprocessed future features table to BigQuery"
        )
        save_table_to_bigquery(
            df_future,
            self.output_database,
            self.preprocessed_output_future_table,
            self.env,
        )

        # Step 2.20 Append counts of valid cars per TCU measurement to BigQuery
        self.logger.info(
            "Step 2.20 Append counts of valid cars per TCU measurement to BigQuery"
        )
        append_table_to_bigquery(
            df_valid_measurements, self.output_database, self.valid_cars, self.env
        )

        # Step 3 Backtest to generate evaluation metrics on current data
        self.logger.info(
            "Step 3 Backtest to generate evaluation metrics on current data"
        )

        # Step 3.1 Prepare training data for odometer forecast
        self.logger.info("Step 3.1 Prepare training data for odometer forecast")
        df_stats_odometer = prepare_features_for_forecaster(
            df_features,
            self.odometer_target,
            self.id_column,
            self.date_column,
            self.static_features,
            self.dynamic_features,
        )

        # Step 3.2 Prepare training data for oil percent forecast
        self.logger.info("Step 3.2 Prepare training data for oil percent forecast")
        df_stats_oil_percent = prepare_features_for_forecaster(
            df_features,
            self.oil_percent_target,
            self.id_column,
            self.date_column,
            self.static_features,
            self.dynamic_features,
        )

        # Step 3.3 Prepare future data
        self.logger.info("Step 3.3 Prepare future data")
        X_future = prepare_future_table(
            df_future, self.id_column, self.date_column, self.dynamic_features
        )

        # Step 3.4 Apply first order differencing to the odometer training data
        self.logger.info(
            "Step 3.4 Apply first order differencing to the odometer training data"
        )
        df_stats_delta_odometer = first_order_difference_target(df_stats_odometer)

        # Step 3.5 Apply first order differencing to the oil percent training data
        self.logger.info(
            "Step 3.5 Apply first order differencing to the oil percent training data"
        )
        df_stats_delta_oil_percent = first_order_difference_target(df_stats_oil_percent)

        # Step 3.6 Backtest gmx_odometer to calculate evaluation metrics for forecast
        self.logger.info(
            "Step 3.6 Backtest gmx_odometer to calculate evaluation metrics for forecast"
        )
        df_backtest_result_odometer = backtest(
            df_stats_delta_odometer,
            self.backtest_horizon,
            self.backtest_n_windows,
            self.xgb_hyperparameters,
            self.lags,
            self.date_features,
            self.static_features,
        )

        # Step 3.7 Backtest gmx_service_oil_interval_percent to calculate evaluation metrics for forecast
        self.logger.info(
            "Step 3.7 Backtest gmx_service_oil_interval_percent to calculate evaluation metrics for forecast"
        )
        df_backtest_result_oil_percent = backtest(
            df_stats_delta_oil_percent,
            self.backtest_horizon,
            self.backtest_n_windows,
            self.xgb_hyperparameters,
            self.lags,
            self.date_features,
            self.static_features,
        )

        # Step 3.8 Postprocess backtest results of gmx_odometer
        self.logger.info("Step 3.8 Postprocess backtest results of gmx_odometer")
        df_backtest_result_odometer_post = postprocess_backtest(
            df_backtest_result_odometer,
            df_stats_odometer,
            self.dynamic_features,
            self.postprocess_dict,
        )

        # Step 3.9 Postprocess backtest results of gmx_service_oil_interval_percent
        self.logger.info(
            "Step 3.9 Postprocess backtest results of gmx_service_oil_interval_percent"
        )
        df_backtest_result_oil_percent_post = postprocess_backtest(
            df_backtest_result_oil_percent,
            df_stats_oil_percent,
            self.dynamic_features,
            self.postprocess_dict,
        )

        # Step 3.10 Calculate evaluation metrics for backtested gmx_odometer results
        self.logger.info(
            "Step 3.10 Calculate evaluation metrics for backtested gmx_odometer results"
        )
        df_odometer_metrics = calculate_evaluation_metrics(
            df_backtest_result_odometer_post,
            df_stats_odometer,
        )

        # Step 3.11 Calculate evaluation metrics for backtested gmx_service_oil_interval_percent results
        self.logger.info(
            "Step 3.11 Calculate evaluation metrics for backtested gmx_service_oil_interval_percent results"
        )
        df_oil_percent_metrics = calculate_evaluation_metrics(
            df_backtest_result_oil_percent_post,
            df_stats_oil_percent,
        )

        # Step 3.12 Save backtest forecast metrics to BigQuery.
        self.logger.info("Step 3.12 Save backtest forecast metrics to BigQuery")
        if self.env == "main":
            append_table_to_bigquery(
                df_odometer_metrics,
                self.output_database,
                f"odometer_{self.metrics_output_table}",
                self.env,
            )
            append_table_to_bigquery(
                df_oil_percent_metrics,
                self.output_database,
                f"oil_percent_{self.metrics_output_table}",
                self.env,
            )
        else:
            save_table_to_bigquery(
                df_odometer_metrics,
                self.output_database,
                f"odometer_{self.metrics_output_table}",
                self.env,
            )
            save_table_to_bigquery(
                df_oil_percent_metrics,
                self.output_database,
                f"oil_percent_{self.metrics_output_table}",
                self.env,
            )

        # Step 4 Forecast future TCU values.
        self.logger.info("Step 4 Forecast future TCU values.")

        # Step 4.1 Build odometer forecaster
        self.logger.info("Step 4.1 Build odometer forecasters")
        odometer_forecasters = build_forecasters(
            df_stats_delta_odometer,
            self.horizon,
            self.n_windows_list,
            self.xgb_hyperparameters,
            self.lags,
            self.date_features,
            self.static_features,
        )

        # Step 4.2 Build oil percent forecaster
        self.logger.info("Step 4.2 Build oil percent forecasters")
        oil_percent_forecasters = build_forecasters(
            df_stats_delta_oil_percent,
            self.horizon,
            self.n_windows_list,
            self.xgb_hyperparameters,
            self.lags,
            self.date_features,
            self.static_features,
        )

        # Step 4.3 Execute odometer forecasters
        self.logger.info("Step 4.3 Execute odometer forecasters")
        odometer_forecast_results = execute_forecasters(
            odometer_forecasters,
            self.n_windows_list,
            X_future,
            self.horizon,
            level=20,
        )

        # Step 4.4 Execute oil percent forecasters
        self.logger.info("Step 4.4 Execute oil percent forecasters")
        oil_percent_forecast_results = execute_forecasters(
            oil_percent_forecasters,
            self.n_windows_list,
            X_future,
            self.horizon,
            level=20,
        )

        # Step 4.5 Apply postprocessing to odometer forecast
        self.logger.info("Step 4.5 Apply postprocessing to odometer forecast")
        df_odometer_forecast_results_post = postprocess_forecast(
            odometer_forecast_results,
            X_future,
            df_stats_odometer,
            self.dynamic_features,
            self.postprocess_dict,
            trend="up",
        )

        # Step 4.6 Apply postprocessing to oil percent forecast
        self.logger.info("Step 4.6 Apply postprocessing to oil percent forecast")
        df_oil_percent_forecast_results_post = postprocess_forecast(
            oil_percent_forecast_results,
            X_future,
            df_stats_oil_percent,
            self.dynamic_features,
            self.postprocess_dict,
            trend="down",
        )

        # Step 4.7 Save postprocessed forecasts
        self.logger.info("Step 4.7 Save postprocessed forecasts")
        if self.env == "main":
            append_table_to_bigquery(
                df_odometer_forecast_results_post,
                self.output_database,
                "odometer_forecast_results",
                self.env,
            )
            append_table_to_bigquery(
                df_oil_percent_forecast_results_post,
                self.output_database,
                "oil_percent_forecast_results",
                self.env,
            )
        else:
            save_table_to_bigquery(
                df_odometer_forecast_results_post,
                self.output_database,
                "odometer_forecast_results",
                self.env,
            )
            save_table_to_bigquery(
                df_oil_percent_forecast_results_post,
                self.output_database,
                "oil_percent_forecast_results",
                self.env,
            )

        # Step 5 Calculate outputs from forecast
        self.logger.info("Step 5 Calculate outputs from forecast")

        # Step 5.1 Calculate dates of next needed oil changes from oil percent forecast
        self.logger.info(
            "Step 5.1 Calculate dates of next needed oil changes from oil percent forecast"
        )
        df_upcoming_oil_percent_service_dates = predict_oil_change_date_from_forecast(
            df_oil_percent_forecast_results_post
        )

        # Step 5.2 Calculate distance travelled since last gmx_service_vehicle_interval_distance value was recorded for each car
        self.logger.info(
            "Step 5.2 Calculate distance travelled since last gmx_service_vehicle_interval_distance value was recorded for each car"
        )
        df_vehicle_service_latest_distance = calculate_latest_service_distances(
            df_features,
            self.vehicle_service_distance_column,
        )

        # Step 5.3 Calculate distance travelled since last gmx_service_oil_interval_distance value was recorded for each car
        self.logger.info(
            "Step 5.3 Calculate distance travelled since last gmx_service_oil_interval_distance value was recorded for each car"
        )
        df_oil_service_latest_distance = calculate_latest_service_distances(
            df_features,
            self.oil_service_distance_column,
        )

        # Step 5.4 Estimate future gmx_service_vehicle_interval_distance using gmx_odometer forecast
        self.logger.info(
            "Step 5.4 Estimate future gmx_service_vehicle_interval_distance using gmx_odometer forecast"
        )
        df_forecasted_vehicle_service_distance = estimate_future_service_distance(
            df_odometer_forecast_results_post,
            df_vehicle_service_latest_distance,
            self.vehicle_service_distance_column,
        )

        # Step 5.5 Estimate future gmx_service_oil_interval_distance using gmx_odometer forecast
        self.logger.info(
            "Step 5.5 Estimate future gmx_service_oil_interval_distance using gmx_odometer forecast"
        )
        df_forecasted_oil_service_distance = estimate_future_service_distance(
            df_odometer_forecast_results_post,
            df_oil_service_latest_distance,
            self.oil_service_distance_column,
        )

        # Step 5.6 Save forecasts for service distances calculated from gmx_odometer forecast
        self.logger.info(
            "Step 5.6 Save forecasts for service distances calculated from gmx_odometer forecast"
        )
        if self.env == "main":
            append_table_to_bigquery(
                df_forecasted_vehicle_service_distance,
                self.output_database,
                "forecasted_vehicle_service_distance",
                self.env,
            )
            append_table_to_bigquery(
                df_forecasted_oil_service_distance,
                self.output_database,
                "forecasted_oil_service_distance",
                self.env,
            )
        else:
            save_table_to_bigquery(
                df_forecasted_vehicle_service_distance,
                self.output_database,
                "forecasted_vehicle_service_distance",
                self.env,
            )
            save_table_to_bigquery(
                df_forecasted_oil_service_distance,
                self.output_database,
                "forecasted_oil_service_distance",
                self.env,
            )

        # Step 5.7 Calculate upcoming vehicle services
        self.logger.info("Step 5.7 Calculate upcoming vehicle services")
        df_upcoming_vehicle_service_distance_dates = get_upcoming_service_dates(
            df_forecasted_vehicle_service_distance,
            f"{self.vehicle_service_distance_column}_estimate",
        )

        # Step 5.8 Calculate upcoming oil services
        self.logger.info("Step 5.8 Calculate upcoming oil services")
        df_upcoming_oil_distance_service_dates = get_upcoming_service_dates(
            df_forecasted_oil_service_distance,
            f"{self.oil_service_distance_column}_estimate",
        )

        # Step 5.9 Concat upcoming oil service predictions
        self.logger.info("Step 5.9 Concat upcoming oil service predictions")
        df_upcoming_oil_service_dates = concat_oil_service_predictions(
            df_upcoming_oil_distance_service_dates,
            df_upcoming_oil_percent_service_dates,
        )

        # Step 5.10 Get last serviceable date for vehicle service
        self.logger.info("Step 5.10 Get last serviceable date for vehicle service")
        df_last_vehicle_service_opportunity_dates = get_last_serviceable_date(
            df_upcoming_vehicle_service_distance_dates, X_future
        )

        # Step 5.11 Get last serviceable date for oil service
        self.logger.info("Step 5.11 Get last serviceable date for oil service")
        df_last_oil_service_opportunity_dates = get_last_serviceable_date(
            df_upcoming_oil_service_dates, X_future
        )

        # Step 5.12 Prepare prediction output tables
        self.logger.info("Step 5.12 Prepare prediction output tables")
        df_vehicle_service_output, df_oil_service_output = prepare_output_tables(
            df_last_vehicle_service_opportunity_dates,
            df_last_oil_service_opportunity_dates,
        )

        # Step 5.13 Save upcoming vehicle services predictions to BigQuery
        self.logger.info(
            "Step 5.13 Save upcoming vehicle services predictions to BigQuery"
        )
        append_table_to_bigquery(
            df_vehicle_service_output,
            self.output_database,
            "upcoming_vehicle_services",
            self.env,
        )

        # Step 5.14 Save upcoming oil services predictions to BigQuery
        self.logger.info("Step 5.14 Save upcoming oil services predictions to BigQuery")
        append_table_to_bigquery(
            df_oil_service_output,
            self.output_database,
            "upcoming_oil_services",
            self.env,
        )

        # Step 6 Generate model monitoring outputs
        self.logger.info("Step 6 Generate model monitoring outputs")

        # Step 6.1 Detect zero crossings in the service columns
        self.logger.info("Step 6.1 Detect zero crossings in the service columns")
        df_zero_crossings = detect_zero_crossings_in_service_columns(
            df_features, self.id_column, self.date_column, self.monitoring_columns
        )
        # Report the oil percent ground-truth copy under its real column name
        df_zero_crossings["service_type"] = df_zero_crossings["service_type"].replace(
            f"{self.oil_percent_target}_ground_truth_state", self.oil_percent_target
        )

        # Step 6.2 Detect resets in the service columns
        self.logger.info("Step 6.2 Detect resets in the service columns")
        df_resets = detect_resets_in_service_columns(
            df_features, self.id_column, self.date_column, self.monitoring_columns
        )
        # Report the oil percent ground-truth copy under its real column name
        df_resets["service_type"] = df_resets["service_type"].replace(
            f"{self.oil_percent_target}_ground_truth_state", self.oil_percent_target
        )

        # Step 6.3 Save zero crossings table to BigQuery
        self.logger.info("Step 6.3 Save zero crossings table to BigQuery")
        save_table_to_bigquery(
            df_zero_crossings,
            self.monitoring_database,
            self.zero_crossings_table,
            self.env,
        )

        # Step 6.4 Save resets table to BigQuery
        self.logger.info("Step 6.4 Save resets table to BigQuery")
        save_table_to_bigquery(
            df_resets,
            self.monitoring_database,
            self.resets_table,
            self.env,
        )

        # Pipeline finished
        self.logger.info("Maintenance forecast pipeline complete")
