import multiprocessing
import pytorch_lightning as pl

from TorchCNN.DataModule import DataModule
from TorchCNN.CNNClassifier import CNNClassifier
from TorchCNN.CNNet import CNNet

def main():
    batch_size = 512
    dataset_path = "dataset/concat"

    model = CNNet()
    classifier = CNNClassifier(model)
    dm = DataModule(batch_size, dataset_path)
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(check_val_every_n_epoch=1, num_sanity_val_steps=0, accelerator="auto", max_epochs=5000)

    trainer.fit(model=classifier, datamodule=dm)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()