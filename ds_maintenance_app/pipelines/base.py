from abc import ABC, abstractmethod
import logging
from datetime import datetime

from ds_maintenance_app.utils.utils import load_config


class BasePipeline(ABC):

    def __init__(self, env: str, pipeline_name: str):
        self.config = load_config("config.yaml")
        self.env = env
        self.pipeline_name = pipeline_name

        # Setup the logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(pipeline_name)

        # Create tag for saved results and artifacts
        now = datetime.now()
        self.datetime_tag = now.strftime("%Y_%m_%d__%H_%M_%S")

    @abstractmethod
    def run_pipeline(self):
        pass
