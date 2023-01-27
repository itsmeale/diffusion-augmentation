import numpy as np
import torch
from loguru import logger
from sklearn.metrics import auc, classification_report, precision_recall_curve
from torch.utils.data import DataLoader


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


def log_experiment_report(model, test_dataset, exp_logger):

    logger.warning(f"Starting evaluation - test set samples = {len(test_dataset)}")

    labels, scores = make_predictions(model, test_dataset)
    y_pred = np.where(np.array(scores) >= 0.5, 1, 0)

    classifier_report = classification_report(labels, y_pred)
    exp_logger.log_text("Classification Report", classifier_report)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    auc_precision_recall = str(auc(recall, precision))

    exp_logger.log_text("Precision-Recall-AUC", auc_precision_recall)
