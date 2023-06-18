# -*- coding: utf-8 -*-
from functools import partial
from typing import Tuple, Dict

import hydra
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from src.dataset.mhist import MHISTDataset
from src.dataset.xray import XRayDataset
from src.experiments.evaluation import log_experiment_report
from src.experiments.log import ExperimentLogger
from src.experiments.train import TrainTestLoop
from src.model import ConvNet


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


def load_dataset(
    train_path: str,
    test_path: str,
    dataset_name: str,
    dataset_resolution: int,
    normalize_params: Tuple,
    train_batch: int,
    test_batch: int,
):
    logger.info("Loading training and test data")

    available_datasets = {
        "MHIST": load_mhist_dataset,
        "XRAY": partial(
            load_xray_dataset,
            train_path=train_path,
            test_path=test_path,
            dataset_resolution=dataset_resolution,
            normalize_params=normalize_params,
        ),
    }

    dataset_loader = available_datasets.get(dataset_name)

    if not dataset_loader:
        raise ValueError(f"Dataset {dataset_name} not available.")

    train_dataset, test_dataset = dataset_loader()

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True)

    return train_dataset, test_dataset, train_loader, test_loader


def load_xray_dataset(train_path, test_path, dataset_resolution, normalize_params):
    train_dataset = XRayDataset(
        root_dir=train_path,
        image_resolution=dataset_resolution,
        normalize_params=normalize_params,
    )
    test_dataset = XRayDataset(
        root_dir=test_path,
        image_resolution=dataset_resolution,
        normalize_params=normalize_params,
    )
    return train_dataset, test_dataset


def load_mhist_dataset():
    train_dataset = MHISTDataset(partition="train")
    test_dataset = MHISTDataset(partition="test")
    return test_dataset, train_dataset


def train(experiment_cfg: Dict):
    # experiment metadata
    EXP_NAME = experiment_cfg.name
    USE_CUDA = experiment_cfg.use_cuda
    # model metadata
    LR = experiment_cfg.model.learning_rate
    TRAIN_BATCH_SIZE = experiment_cfg.model.train_batch_size
    TEST_BATCH_SIZE = experiment_cfg.model.test_batch_size
    EPOCHS = experiment_cfg.model.epochs
    INPUT_DROPOUT = experiment_cfg.model.input_dropout
    DENSE_DROPOUT = experiment_cfg.model.dense_dropout
    MAX_EPOCHS_WITHOUT_IMPROVEMENT = experiment_cfg.model.max_epochs_without_improvement
    # dataset metadata
    DATASET_NAME = experiment_cfg.dataset.name
    DATASET_RESOLUTION = experiment_cfg.dataset.image_resolution
    NORMALIZE_PARAMS = (
        experiment_cfg.dataset.normalization_mean,
        experiment_cfg.dataset.normalization_std,
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

    # setup experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=f"runs/{EXP_NAME}", hparams_dict=hparams_dict
    )

    train_dataset, test_dataset, train_loader, test_loader = load_dataset(
        train_path=experiment_cfg.dataset.train_path,
        test_path=experiment_cfg.dataset.test_path,
        dataset_name=DATASET_NAME,
        dataset_resolution=DATASET_RESOLUTION,
        normalize_params=NORMALIZE_PARAMS,
        train_batch=TRAIN_BATCH_SIZE,
        test_batch=TEST_BATCH_SIZE,
    )

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

    TrainTestLoop(
        model,
        loss_fn,
        optimizer,
        train_dataset,
        test_dataset,
        epochs=EPOCHS,
        experiment_logger=exp_logger,
        max_epochs_without_improvement=MAX_EPOCHS_WITHOUT_IMPROVEMENT,
        batch_size=TRAIN_BATCH_SIZE,
    ).run()

    log_experiment_report(experiment_cfg.name, model, test_dataset, exp_logger)


@hydra.main(version_base=None, config_path="../conf", config_name="experiments")
def main(cfg):
    for experiment_cfg in cfg.experiments:
        for i in range(50):
            train(experiment_cfg=experiment_cfg)



if __name__ == "__main__":
    main()
