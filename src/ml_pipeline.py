import logging
import mlflow
from matplotlib import pyplot as plt

from mlops.src.ModelWrapper import ModelWrapper
from mlops.src.MLFlowManager import MLFlowManager
from src.training_orchestrator import ModelOrchestrator
import torch

class MLPipeline:
    def __init__(self, orchestrator: ModelOrchestrator, mlflow_manager: MLFlowManager, device=None):
        self.orchestrator = orchestrator
        self.mlflow_manager = mlflow_manager
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.orchestrator.model.to(self.device)  # Move the model to the selected device
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def run_training_pipeline(
            self, data_paths, filename, batch_size, loss_function, optimizer, epochs, model_name
    ):
        processed_path, raw_path = data_paths
        self.logger.info("Starting data preparation...")
        self.orchestrator.prepare_data(processed_path, raw_path, filename)
        self.logger.info("Loading training and testing data...")
        train_loader, test_loader = self.orchestrator.get_data_loaders(processed_path, filename, batch_size)
        self.logger.info("Starting MLflow experiment tracking...")
        with mlflow.start_run() as run:
            self.mlflow_manager.log_params({"batch_size": batch_size, "epochs": epochs})
            self.logger.info("Starting training process...")
            for epoch in range(epochs):
                train_loss = self.orchestrator.train_model(train_loader, loss_function, optimizer, self.device)
                eval_loss = self.orchestrator.eval_model(test_loader, loss_function, self.device)
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {eval_loss:.4f}")
                self.mlflow_manager.log_metrics({"train_loss": train_loss, "test_loss": eval_loss, "epoch": epoch + 1},
                                                step=epoch)
            self.logger.info("Training complete. Logging the model...")
            model_wrapper = ModelWrapper(self.orchestrator.model)
            full_eval_loss = self.run_evaluation_pipeline(data_paths, filename, batch_size, loss_function)
            predictions = self.run_prediction_pipeline(data_paths, filename, batch_size)
            self.mlflow_manager.log_metrics({"full_eval_loss": full_eval_loss})
            self.mlflow_manager.log_model(model_wrapper, model_name, run=run)
        self.logger.info("Training pipeline complete.")
        return train_loader, test_loader

    def run_evaluation_pipeline(self, data_paths, filename, batch_size, loss_function):
        processed_path, _ = data_paths
        self.logger.info("Loading evaluation data...")
        eval_loader = self.orchestrator.get_eval_loader(processed_path, filename, batch_size)
        self.logger.info("Evaluating the model...")
        eval_loss = self.orchestrator.eval_model(eval_loader, loss_function, self.device)
        self.logger.info(f"Evaluation Loss: {eval_loss:.4f}")
        return eval_loss

    def run_prediction_pipeline(self, data_paths, filename, batch_size):
        processed_path, _ = data_paths
        self.logger.info("Loading data for prediction...")
        eval_loader = self.orchestrator.get_eval_loader(processed_path, filename, batch_size)
        self.logger.info("Generating predictions...")
        predictions = self.orchestrator.predict(eval_loader, device=self.device)
        eval_dataset = self.orchestrator.get_eval_dataset(processed_path, filename)
        self.generate_predictions_plot(eval_dataset.y, predictions)
        return predictions

    def generate_predictions_plot(self, y_actual, y_predicted):
        num_outputs = y_actual.shape[1]
        num_rows = (num_outputs + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        for i in range(num_outputs):
            axes[i].plot(y_actual[:, i], label="Actual")
            axes[i].plot(y_predicted[:, i], label="Predicted")
            axes[i].set_title(f"y{i + 1}")
            axes[i].legend()
        fig.tight_layout()
        self.mlflow_manager.log_fig(fig, "predictions_plot.png")
        return fig
