import torch
from sklearn.metrics import classification_report


def make_classification_report(model, test_loader):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs).detach().cpu().numpy()
            labels = labels.cpu().numpy()

            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

            all_labels += list(labels)
            all_preds += list(preds)

    return classification_report(all_labels, all_preds)


def log_experiment_report(model, test_loader, exp_logger):
    classifier_report = make_classification_report(model, test_loader)
    exp_logger.log_text("Classification Report", classifier_report)
