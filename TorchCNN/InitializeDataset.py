import os
import re
import hashlib
import json
import random
from pathlib import Path
from TorchCNN.CNNDataset import CNNDataset

class InitializeDataset():
    def __init__(self, dataset_path):
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
        for subdir, dirs, files in os.walk(self.dataset_path):
            self.x.extend([os.path.join(subdir, f) for f in files if f.lower().endswith(".npy")])
        with open(os.path.join(self.dataset_path, "one_hot_encoding.json")) as json_file:
            self.one_hot_dict = json.load(json_file)

        random.shuffle(self.x)

    def split_dataset(self, validation_percentage, testing_percentage):
        for item in self.x:
            subset = self.which_set(item, validation_percentage, testing_percentage)
            label = item.split(os.sep)[2]
            one_hot = self.one_hot_dict[label]
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
        return CNNDataset(self.y_train, self.x_train, transform)

    def validate_dataset(self, transform=None):
        return CNNDataset(self.y_validate, self.x_validate, transform)

    def test_dataset(self, transform=None):
        return CNNDataset(self.y_test, self.x_test, transform)