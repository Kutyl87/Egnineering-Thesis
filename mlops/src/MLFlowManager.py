import logging
import stat

import mlflow
import mlflow.pyfunc
import os
from mlops.src.ModelWrapper import ModelWrapper


class MLFlowManager:
    def __init__(self, experiment_name: str, tracking_uri=None) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info(f"Tracking URI set to: {tracking_uri}")
        else:
            self.logger.warning("No tracking URI provided. Defaulting to local.")
        self.experiment_name = experiment_name
        self.logger.info(f"Experiment set to: {experiment_name}")
        mlflow.set_experiment(experiment_name=experiment_name)

    def log_model(self, model: ModelWrapper, model_name: str, run: any) -> None:
        # with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(artifact_path=model_name, python_model=model)
        self.logger.info(
            f"Model '{model_name}' logged under run ID: {run.info.run_id}"
        )

    def log_artifacts(self, artifact_path: str) -> None:
        mlflow.log_artifacts(artifact_path)
        self.logger.info(f"Artifacts logged from path: {artifact_path}")

    def log_metrics(self, metrics: dict, step=None) -> None:
        if step is not None:
            mlflow.log_metrics(metrics, step=step)
            self.logger.info(f"Metrics logged: {metrics} at step: {step}")
        else:
            mlflow.log_metrics(metrics)
            self.logger.info(f"Metrics logged: {metrics}")

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)
        self.logger.info(f"Parameters logged: {params}")

    def load_model(self, path: str, model_name: str):
        model_uri = f"{path}/{model_name}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            self.logger.info(f"Model '{model_name}' loaded from {model_uri}")
            return model
        except Exception as e:
            self.logger.error(
                f"Failed to load model '{model_name}' from '{model_uri}': {e}"
            )

    def log_fig(self, fig, fig_name: str) -> None:
        mlflow.log_figure(fig, fig_name)

