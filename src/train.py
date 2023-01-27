# -*- coding: utf-8 -*-
""" Treina uma rede convolucional e avalia
nos respectivos conjuntos de treino e teste.

Ao final, salva os resultados na diretório
de experimentos do projeto.
"""

from pathlib import Path
from time import time
from typing import Dict, Union

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset.mhist import MHISTDataset
from src.dataset.xray import XRayDataset
from src.model import ConvNet


class ExperimentLogger:
    def __init__(self, experiment_name, hparams_dict):
        self.writer = SummaryWriter(experiment_name)
        self.hparam_dict = hparams_dict

    def log_metric(self, metric_name: str, values: Union[Dict, float], epoch: int):
        if not isinstance(values, dict):
            self.writer.add_scalar(metric_name, values, epoch)

        self.writer.add_scalars(
            metric_name,
            values,
            epoch,
        )

    def log_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def log_hparams(self, metrics_dict: Dict):
        self.writer.add_hparams(hparam_dict=self.hparam_dict, metric_dict=metrics_dict)

    def log_embedding(self, data):
        self.writer.add_embedding(data)

    def log_images(self, images):
        self.writer.add_images("Samples", images)


class TrainTestLoop:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        epochs,
        experiment_logger,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.experiment_logger = experiment_logger

    def train_loop(self):
        running_loss = 0
        running_acc = 0
        n = 0
        self.model.train()

        for images, targets in self.train_loader:
            pred = self.model.forward(images)

            loss = self.loss_fn(pred, targets.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += self.calculate_accuracy(pred, targets)
            n += 1

        return running_loss / n, running_acc / n

    def calculate_accuracy(self, pred, targets):
        pred = pred.detach().flatten().cpu().numpy()
        targets = targets.cpu().numpy()

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        return np.sum((pred == targets)) / len(targets)

    def test_loop(self):
        self.model.eval()
        running_loss = 0
        running_acc = 0
        n = 0

        self.optimizer.zero_grad()

        with torch.no_grad():
            for images, targets in self.val_loader:
                pred = self.model(images)
                val_loss = self.loss_fn(pred, targets.unsqueeze(1))
                running_loss += val_loss.item()
                running_acc += self.calculate_accuracy(pred, targets)
                n += 1

        return running_loss / n, running_acc / n

    def run(self):
        logger.info("Starting train test loop")
        train_loss = []
        test_loss = []

        for i in range(self.epochs):
            _train_loss, _train_acc = self.train_loop()
            _test_loss, _test_acc = self.test_loop()
            train_loss.append(_train_loss)
            test_loss.append(_test_loss)

            self.experiment_logger.log_metric(
                metric_name="loss",
                values={"train": _train_loss, "test": _test_loss},
                epoch=i,
            )

            logger.info(f"Epoch {i} - train acc: {_train_acc} - test acc: {_test_acc}")

        self.experiment_logger.log_hparams(
            {
                "train_acc": _train_acc,
                "test_acc": _test_acc,
                "train_loss": _train_loss,
                "val_loss": _test_loss,
            }
        )

        # self.experiment_logger.log_graph(self.model, images)
        # self.experiment_logger.log_embedding(images)

        return train_loss, test_loss


def make_hparam_dict(
    epochs,
    lr,
    train_batch_size,
    input_dropout,
    dense_dropout,
    image_resolution,
    normalization_mean,
    normalization_std,
):
    return {
        "epochs": epochs,
        "learning_rate": lr,
        "train_batch_size": train_batch_size,
        "input_dropout": input_dropout,
        "dense_dropout": dense_dropout,
        "image_resolution": image_resolution,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="experiments")
def train(cfg):
    # experiment metadata
    EXP_NAME = cfg.experiment.name
    EXP_VERSION = cfg.experiment.version
    USE_CUDA = cfg.experiment.use_cuda
    # model metadata
    LR = cfg.experiment.model.learning_rate
    TRAIN_BATCH_SIZE = cfg.experiment.model.train_batch_size
    TEST_BATCH_SIZE = cfg.experiment.model.test_batch_size
    EPOCHS = cfg.experiment.model.epochs
    INPUT_DROPOUT = cfg.experiment.model.input_dropout
    DENSE_DROPOUT = cfg.experiment.model.dense_dropout
    # dataset metadata
    DATASET_NAME = cfg.experiment.dataset.name
    DATASET_RESOLUTION = cfg.experiment.dataset.image_resolution
    NORMALIZE_PARAMS = (
        cfg.experiment.dataset.normalization_mean,
        cfg.experiment.dataset.normalization_std,
    )

    hparams_dict = make_hparam_dict(
        EPOCHS,
        LR,
        TRAIN_BATCH_SIZE,
        INPUT_DROPOUT,
        DENSE_DROPOUT,
        DATASET_RESOLUTION,
        NORMALIZE_PARAMS[0],
        NORMALIZE_PARAMS[1],
    )

    logger.info("Loading training and test data")

    # setup dataset
    if DATASET_NAME not in {"MHIST", "XRAY"}:
        raise ValueError(f"{DATASET_NAME} not available")

    if DATASET_NAME == "MHIST":
        train_dataset = MHISTDataset(partition="train")
        test_dataset = MHISTDataset(partition="test")
    elif DATASET_NAME == "XRAY":
        train_dataset = XRayDataset(
            root_dir="data/raw/xray/train",
            image_resolution=DATASET_RESOLUTION,
            normalize_params=NORMALIZE_PARAMS,
        )
        test_dataset = XRayDataset(
            root_dir="data/raw/xray/test",
            image_resolution=DATASET_RESOLUTION,
            normalize_params=NORMALIZE_PARAMS,
        )

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

    # setup model
    model = ConvNet(input_dropout=INPUT_DROPOUT, dense_dropout=DENSE_DROPOUT)

    # load to GPU
    if USE_CUDA and torch.cuda.is_available():
        logger.info("Loading data and model to GPU")
        train_dataset.to_cuda()
        test_dataset.to_cuda()
        model.to("cuda")

    # setup model optimizer and loss
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.BCELoss()

    # setup experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=f"runs/{EXP_NAME}", hparams_dict=hparams_dict
    )

    loader = DataLoader(train_dataset[:25], batch_size=25)
    to_log_images, _ = next(iter(loader))
    print(to_log_images.shape)

    exp_logger.log_graph(model, to_log_images)
    exp_logger.log_images(to_log_images)

    train_loop = TrainTestLoop(
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        epochs=EPOCHS,
        experiment_logger=exp_logger,
    )
    train_loss, test_loss = train_loop.run()

    make_experiment_report(
        model, test_loader, train_loss, test_loss, EXP_NAME, EXP_VERSION
    )

    return train_loss, test_loss


def make_classification_report(model, test_loader):
    all_labels = list()
    all_preds = list()

    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs).detach().cpu().numpy()
            labels = labels.cpu().numpy()

            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

            all_labels += list(labels)
            all_preds += list(preds)

    return classification_report(all_labels, all_preds)


def make_experiment_report(
    model, test_loader, train_loss, test_loss, exp_name, exp_version
):
    now = str(int(time()))
    path = Path(f"experiments/{exp_name}-{exp_version}-{now}/")
    path.mkdir(exist_ok=True)

    loss_df = pd.DataFrame({"train_loss": train_loss, "test_loss": test_loss})
    loss_df.to_csv(path.joinpath("loss.csv"), index=False)

    classifier_report = make_classification_report(model, test_loader)
    with open(path.joinpath("classification_report.txt"), "w") as cfile:
        cfile.write(classifier_report)


if __name__ == "__main__":
    train()
