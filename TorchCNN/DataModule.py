import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from TorchCNN.InitializeDataset import InitializeDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset_dir, val_test_ratio, other_test_only):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.val_test_ratio = val_test_ratio
        self.other_test_only = other_test_only

    def prepare_data(self):
        validation_percentage, testing_percentage = self.val_test_ratio
        self.dataset = InitializeDataset(self.dataset_dir)
        self.dataset.initialize_data()
        self.dataset.split_dataset(validation_percentage, testing_percentage, self.other_test_only)


    def setup(self, stage=None):
        self.train_dataset = self.dataset.train_dataset()
        self.validate_dataset = self.dataset.validate_dataset()
        self.test_dataset = self.dataset.test_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.validate_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)


