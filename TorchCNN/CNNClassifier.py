from typing import Any
import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from TorchCNN.Metrics import Metrics


class CNNClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating a CNN model.

    Args:
        model: A PyTorch neural network model to be trained and evaluated.

    Attributes:
        model (nn.Module): The CNN model used for predictions.
        lr (float): The learning rate for the optimizer.
        size (tuple): Input size for the model.
        sigma (float): A parameter for any additional operations (if used).
        out_features (int): Number of output features from the model.
        metrics (Metrics): A custom metrics object to compute evaluation metrics.
        loss_fn (nn.Module): Loss function used for training and evaluation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 0.001
        self.size = (20, 96)
        self.sigma = 3.3
        self.out_features = model.out_features
        self.metrics = Metrics(self.out_features).to(device=self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model(x)

    def common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            tuple: Loss, outputs, and true labels.
        """
        x, y = batch
        outputs = self(x)
        y_indices = torch.argmax(y, dim=1)
        loss = self.loss_fn(outputs, y_indices)
        return loss, outputs, y

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Training step to calculate and log the loss.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def common_test_valid_step(self, batch, batch_idx):
        """
        Common step for validation and testing.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.metrics = self.metrics.to(device=self.device)
        self.metrics.update(outputs, y)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """
        Validation step to calculate and log the validation loss.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log validation metrics at the end of the epoch.

        Returns:
            None
        """
        results = self.metrics.compute()
        self.log("val_mean_acc", results["mean_accuracy"], prog_bar=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx) -> None:
        """
        Test step to calculate and log the test loss.

        Args:
            batch (tuple): A tuple containing inputs and labels.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """
        Compute and log test metrics at the end of the epoch.

        Returns:
            None
        """
        results = self.metrics.compute()
        self.log("test_mean_acc", results["mean_accuracy"], prog_bar=True)
        self.metrics.reset()

    def configure_optimizers(self):
        """
        Configure and return the optimizer.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
