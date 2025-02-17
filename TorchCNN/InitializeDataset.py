import os
import re
import hashlib
import json
import random
from pathlib import Path
from TorchCNN.CNNDataset import CNNDataset

class InitializeDataset:
    """
    A class to initialize and process a dataset for training, validation, and testing.

    Attributes:
        dataset_path (Path): Path to the dataset directory.
        x (list): List of all dataset file paths.
        one_hot_dict (dict): Dictionary for one-hot encoding of labels.
        x_train (list): File paths for the training set.
        x_validate (list): File paths for the validation set.
        x_test (list): File paths for the testing set.
        y_train (list): Labels for the training set.
        y_validate (list): Labels for the validation set.
        y_test (list): Labels for the testing set.
        MAX_NUM_WAVS_PER_CLASS (int): Maximum number of WAV files per class, used for hashing.
    """

    def __init__(self, dataset_path):
        """
        Initialize the dataset.

        Args:
            dataset_path (str): Path to the dataset directory.
        """
        self.dataset_path = Path(dataset_path)
        self.x = []
        self.one_hot_dict = {}
        self.x_train = []
        self.x_validate = []
        self.x_test = []
        self.y_train = []
        self.y_validate = []
        self.y_test = []

        self.MAX_NUM_WAVS_PER_CLASS = 2**27 - 1

    def which_set(self, filename, validation_percentage, testing_percentage):
        """
        Function from the Dataset. Determines which subset (training, validation, or testing) a file belongs to based on a hash of its name.

        Args:
            filename (str): Path to the file.
            validation_percentage (float): Percentage of files to include in the validation set.
            testing_percentage (float): Percentage of files to include in the testing set.

        Returns:
            str: Subset label ('training', 'validation', or 'testing').
        """
        base_name = os.path.basename(filename)
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (self.MAX_NUM_WAVS_PER_CLASS + 1)) *
                           (100.0 / self.MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < validation_percentage:
            result = 'validation'
        elif percentage_hash < (testing_percentage + validation_percentage):
            result = 'testing'
        else:
            result = 'training'
        return result

    def initialize_data(self):
        """
        Initialize the dataset by loading file paths and one-hot encoding.

        Collects all .npy files in the dataset directory and loads the one-hot encoding mapping from a JSON file.
        """
        for subdir, dirs, files in os.walk(self.dataset_path):
            self.x.extend([os.path.join(subdir, f) for f in files if f.lower().endswith(".npy")])
        with open(os.path.join(self.dataset_path, "one_hot_encoding.json")) as json_file:
            self.one_hot_dict = json.load(json_file)

        random.shuffle(self.x)

    def split_dataset(self, validation_percentage, testing_percentage, other_test_only):
        """
        Split the dataset into training, validation, and testing subsets.

        Args:
            validation_percentage (float): Percentage of files to include in the validation set.
            testing_percentage (float): Percentage of files to include in the testing set.
            other_test_only (list): Words to be included only in the testing set in other class.
        """
        if "other" in self.one_hot_dict:
            accepted_words = [key for key in self.one_hot_dict if key != "other"]
        else:
            accepted_words = [key for key in self.one_hot_dict]
            other_test_only = []

        for item in self.x:
            label = item.split(os.sep)[2]
            if label in other_test_only:
                subset = "testing"
            else:
                subset = self.which_set(item, validation_percentage, testing_percentage)

            if label in self.one_hot_dict:
                one_hot = self.one_hot_dict[label]
            else:
                one_hot = self.one_hot_dict["other"]

            if subset == 'training':
                self.x_train.append(item)
                self.y_train.append(one_hot)
            elif subset == 'validation':
                self.x_validate.append(item)
                self.y_validate.append(one_hot)
            elif subset == 'testing':
                self.x_test.append(item)
                self.y_test.append(one_hot)

    def train_dataset(self, transform=None):
        """
        Create a training dataset object.

        Args:
            transform (callable, optional): Optional transform to apply to the data.

        Returns:
            CNNDataset: Dataset object for training.
        """
        return CNNDataset(self.y_train, self.x_train, transform)

    def validate_dataset(self, transform=None):
        """
        Create a validation dataset object.

        Args:
            transform (callable, optional): Optional transform to apply to the data.

        Returns:
            CNNDataset: Dataset object for validation.
        """
        return CNNDataset(self.y_validate, self.x_validate, transform)

    def test_dataset(self, transform=None):
        """
        Create a testing dataset object.

        Args:
            transform (callable, optional): Optional transform to apply to the data.

        Returns:
            CNNDataset: Dataset object for testing.
        """
        return CNNDataset(self.y_test, self.x_test, transform)
