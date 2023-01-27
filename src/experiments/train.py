import copy

import numpy as np
import torch
from loguru import logger


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
        self.best_test_acc = -np.inf
        self.best_state_dict = None
        self.max_epochs_without_improvement = 1000
        self.epochs_without_improvement = 0
        self.force_stop_training = False

    def check_early_stop(self, current_test_acc):
        if current_test_acc > self.best_test_acc:
            self.epochs_without_improvement = 0
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_test_acc = current_test_acc
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement == self.max_epochs_without_improvement:
            self.force_stop_training = True

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

        acc = running_acc / n
        loss = running_loss / n

        self.check_early_stop(acc)

        return loss, acc

    def log_metric(self, metric_name, train_metric, val_metric, epoch):
        self.experiment_logger.log_metric(
            metric_name=metric_name,
            values={
                "train": train_metric,
                "test": val_metric,
            },
            epoch=epoch,
        )

    def run(self):
        logger.info("Starting train test loop")

        for i in range(self.epochs):
            _train_loss, _train_acc = self.train_loop()
            _test_loss, _test_acc = self.test_loop()

            self.log_metric("Accuracy", _train_acc, _test_acc, i)
            self.log_metric("Loss", _train_loss, _test_loss, i)

            logger.info(f"Epoch {i} - train acc: {_train_acc} - test acc: {_test_acc}")

            if self.force_stop_training:
                break

        self.experiment_logger.log_hparams(
            {
                "train_acc": _train_acc,
                "test_acc": self.best_test_acc,
            }
        )

        self.model.load_state_dict(self.best_state_dict)

        return self.model
