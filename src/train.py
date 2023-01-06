# -*- coding: utf-8 -*-
""" Treina uma rede convolucional e avalia
nos respectivos conjuntos de treino e teste.

Ao final, salva os resultados na diretório
de experimentos do projeto.
"""

import os
from pathlib import Path
from time import time

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset.mhist import MHISTDataset
from src.dataset.xray import XRayDataset

from sklearn.metrics import classification_report

from src.model import ConvNet

from loguru import logger


class TrainTestLoop:

    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, epochs):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

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
        
        return running_loss/n, running_acc/n

    def calculate_accuracy(self, pred, targets):
        pred = pred.detach().flatten().cpu().numpy()
        targets = targets.cpu().numpy()

        pred[pred >= .5] = 1
        pred[pred < .5] = 0

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
            
        return running_loss/n, running_acc/n
    
    def run(self):
        logger.info("Starting train test loop")
        train_loss = []
        test_loss = []

        for i in range(self.epochs):
            _train_loss, _train_acc = self.train_loop()
            _test_loss, _test_acc = self.test_loop()
            train_loss.append(_train_loss)
            test_loss.append(_test_loss)

            logger.info(f"Epoch {i} - train acc: {_train_acc} - test acc: {_test_acc}")
        
        return train_loss, test_loss


def train():
    logger.info("Loading training and test data")
    train_dataset = MHISTDataset(partition="train").to_cuda()
    test_dataset = MHISTDataset(partition="test").to_cuda()

    #train_dataset = XRayDataset(root_dir="data/xray/train").to_cuda()
    #test_dataset = XRayDataset(root_dir="data/xray/test").to_cuda()

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    model = ConvNet()
    model.to("cuda")

    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=.9)
    loss_fn = nn.BCELoss()

    train_loop = TrainTestLoop(model, loss_fn, optimizer, train_loader, test_loader, epochs=10)
    train_loss, test_loss = train_loop.run()
    
    make_experiment_report(model, test_loader, train_loss, test_loss)

    return train_loss, test_loss


def make_classification_report(model, test_loader):
    all_labels = list()
    all_preds = list()

    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs).detach().cpu().numpy()
            labels = labels.cpu().numpy()

            preds[preds >= .5] = 1
            preds[preds < .5] = 0

            all_labels += list(labels)
            all_preds += list(preds)
    
    return classification_report(all_labels, all_preds)


def make_experiment_report(model, test_loader, train_loss, test_loss):
    now = str(int(time()))
    path = Path(f"experiments/{now}/")
    path.mkdir(exist_ok=True)

    loss_df = pd.DataFrame({"train_loss": train_loss, "test_loss": test_loss})
    loss_df.to_csv(path.joinpath("loss.csv"), index=False)

    classifier_report = make_classification_report(model, test_loader)
    with open(path.joinpath("classification_report.txt"), "w") as cfile:
        cfile.write(classifier_report)


if __name__ == "__main__":
    tloss, tsloss = train()
