import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from TorchCNN.InitializeDataset import InitializeDataset

class DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling data preparation, loading, and splitting.

    Attributes:
        batch_size (int): The batch size to use for the DataLoader.
        dataset_dir (str): The directory where the dataset is located.
        val_test_ratio (tuple): A tuple containing the validation and testing split ratios.
        other_test_only (bool): If True, only the testing dataset is prepared (used for specific scenarios).
    """

    def __init__(self, batch_size, dataset_dir, val_test_ratio, other_test_only):
        """
        Initialize the DataModule with specified parameters.

        Args:
            batch_size (int): The batch size to use for the DataLoader.
            dataset_dir (str): The directory where the dataset is located.
            val_test_ratio (tuple): A tuple (validation_percentage, testing_percentage) defining the split ratios.
            other_test_only (bool): Whether to prepare only the testing dataset.
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.val_test_ratio = val_test_ratio
        self.other_test_only = other_test_only

    def prepare_data(self):
        """
        Prepare the dataset by initializing and splitting it based on the specified validation and testing ratios.

        This method initializes the dataset and runs all additional function that are required by InitializeDataset class.
        """
        validation_percentage, testing_percentage = self.val_test_ratio
        self.dataset = InitializeDataset(self.dataset_dir)
        self.dataset.initialize_data()
        self.dataset.split_dataset(validation_percentage, testing_percentage, self.other_test_only)

    def setup(self, stage=None):
        """
        Set up datasets for different stages (train, validate, test).

        This method assigns the appropriate subsets of the dataset to `train_dataset`, `validate_dataset`, and `test_dataset`.

        Args:
            stage (str, optional): The stage of training ('fit', 'test', etc.). Defaults to None. Argument required by
            pytorchLightning but not used in this function.
        """
        self.train_dataset = self.dataset.train_dataset()
        self.validate_dataset = self.dataset.validate_dataset()
        self.test_dataset = self.dataset.test_dataset()

    def train_dataloader(self):
        """
        Create the DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset with the specified batch size and settings.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        """
        Create the DataLoader for the testing dataset.

        Returns:
            DataLoader: The DataLoader for the testing dataset with the specified batch size and settings.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        """
        Create the DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader for the validation dataset with the specified batch size and settings.
        """
        return DataLoader(self.validate_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)



