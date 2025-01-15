from typing import Any
import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from TorchCNN.Metrics import Metrics


class CNNClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 0.001
        self.size = (20, 96)
        self.sigma = 3.3
        self.out_features = model.out_features
        self.metrics = Metrics(self.out_features).to(device = self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        y_indices = torch.argmax(y, dim=1)
        loss = self.loss_fn(outputs, y_indices)
        return loss, outputs, y

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.metrics = self.metrics.to(device=self.device)
        self.metrics.update(outputs, y)
        return loss


    def validation_step(self, batch, batch_idx) -> None:
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        results = self.metrics.compute()
        self.log("val_mean_acc", results["mean_accuracy"], prog_bar=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx) -> None:
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)