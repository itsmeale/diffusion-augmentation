import copy

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class TrainTestLoop:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_dataset,
        val_dataset,
        test_dataset,
        epochs,
        experiment_logger,
        max_epochs_without_improvement,
        batch_size,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.experiment_logger = experiment_logger
        self.batch_size = batch_size
        self.best_val_f1 = -np.inf
        self.best_state_dict = None
        self.max_epochs_without_improvement = max_epochs_without_improvement
        self.epochs_without_improvement = 0
        self.force_stop_training = False

    def check_early_stop(self, current_test_f1):
        if current_test_f1 > self.best_val_f1:
            self.epochs_without_improvement = 0
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_val_f1 = current_test_f1
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement == self.max_epochs_without_improvement:
            self.force_stop_training = True

    def train_loop(self):
        running_loss = 0
        running_f1 = 0
        n = 0
        self.model.train()

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        for images, targets in train_loader:
            pred = self.model.forward(images)

            loss = self.loss_fn(pred, targets.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            pred, targets = self.get_real_labels_and_predictions(pred, targets)
            running_f1 += self.calculate_f1_score(pred, targets)

            n += 1

        return running_loss / n, running_f1 / n

    def calculate_accuracy(self, pred, targets):
        return np.sum((pred == targets)) / len(targets)

    def calculate_f1_score(self, pred, targets):
        return f1_score(targets, pred, average="macro")

    def get_real_labels_and_predictions(self, pred, targets):
        pred = pred.detach().flatten().cpu().numpy()
        targets = targets.cpu().numpy()

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return targets, pred

    def test_loop(self, use_validation_dataset: bool = True):
        self.model.eval()
        running_loss = 0
        running_f1 = 0
        n = 0

        self.optimizer.zero_grad()

        dataset = self.val_dataset if use_validation_dataset else self.test_dataset
        batch_size = 8 if use_validation_dataset else self.batch_size

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            for images, targets in loader:
                pred = self.model(images)
                val_loss = self.loss_fn(pred, targets.unsqueeze(1))
                running_loss += val_loss.item()

                pred, targets = self.get_real_labels_and_predictions(pred, targets)
                running_f1 += self.calculate_f1_score(pred, targets)
                n += 1

        f1 = running_f1 / n
        loss = running_loss / n

        if use_validation_dataset:
            self.check_early_stop(f1)

        return loss, f1

    def log_metric(self, metric_name, train_metric, val_metric, test_metric, epoch):
        self.experiment_logger.log_metric(
            metric_name=metric_name,
            values={
                "train": train_metric,
                "val": val_metric,
                "test": test_metric,
            },
            epoch=epoch,
        )

    def run(self):
        logger.info("Starting train test loop")

        logger.warning(
            f"train set len: {len(self.train_dataset)} - test set len: {len(self.val_dataset)} - test set len: {len(self.test_dataset)}"
        )

        for i in range(self.epochs):
            _train_loss, _train_f1 = self.train_loop()
            _val_loss, _val_f1 = self.test_loop(use_validation_dataset=True)
            _test_loss, _test_f1 = self.test_loop(use_validation_dataset=False)

            self.log_metric("f1_macro", _train_f1, _val_f1, _test_f1, i)
            self.log_metric("loss", _train_loss, _val_loss, _test_loss, i)

            logger.info(f"Epoch {i} - train f1: {_train_f1} - val f1: {_val_f1}")

            if self.force_stop_training:
                break

        self.experiment_logger.log_hparams(
            {
                "train_f1": _train_f1,
                "val_f1": self.best_val_f1,
            }
        )

        self.model.load_state_dict(self.best_state_dict)

        return self.model
