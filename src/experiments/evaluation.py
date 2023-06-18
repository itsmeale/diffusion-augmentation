import pickle
from typing import List

from uuid import uuid4

from dataclasses import dataclass
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import precision_recall_curve, confusion_matrix
from torch.utils.data import DataLoader


@dataclass
class EvalMetrics:
    experiment_name: str

    precision: float
    recall: float
    accuracy: float

    true_positive: int
    true_negative: int

    false_positive: int
    false_negative: int

    precision_recall_curve: List[List[float]]

    y_true: List[int]
    y_pred: List[int]
    scores: List[float]


def make_predictions(model, test_dataset):
    labels = []
    scores = []

    print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    with torch.no_grad():
        for images, y in test_loader:
            _scores = model(images).detach().cpu().flatten().numpy().tolist()
            _labels = y.cpu().flatten().numpy().tolist()

            labels += list(_labels)
            scores += list(_scores)

    logger.warning(f"labels: {len(labels)}")
    logger.warning(f"scores: {len(scores)}")

    return labels, scores


def log_experiment_report(exp_name: str, model, test_dataset, exp_logger):

    logger.warning(f"Starting evaluation - test set samples = {len(test_dataset)}")

    labels, scores = make_predictions(model, test_dataset)
    y_pred = np.where(np.array(scores) >= 0.5, 1, 0)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()

    eval_dataclass = EvalMetrics(
        experiment_name=exp_name,

        precision= tp / (tp+fp),
        recall= tp / (tp+fn),
        accuracy= (tp+tn) / (tp+tn+fp+fn),

        true_positive=tp,
        true_negative=tn,

        false_positive=fp,
        false_negative=fn,

        precision_recall_curve=(precision, recall, thresholds),

        y_true=labels,
        y_pred=y_pred,
        scores=scores,
    )

    with open(f"data/results/eval_metrics/{str(uuid4())}.pkl", "wb") as f:
        pickle.dump(eval_dataclass, f, protocol=pickle.HIGHEST_PROTOCOL)



