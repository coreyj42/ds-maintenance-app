import argparse

from ds_maintenance_app.pipelines.maintenance_forecast_pipeline import (
    MaintenanceForecastPipeline,
)

# Get the parameters from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="dev")
args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    # Run the train pipeline with the parameters from CLI
    train_pipeline = MaintenanceForecastPipeline(env=args.env)
    train_pipeline.run_pipeline()
