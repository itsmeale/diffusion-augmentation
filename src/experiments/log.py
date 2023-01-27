from time import time
from typing import Dict, Union

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, experiment_name, hparams_dict):
        t = int(time())
        self.writer = SummaryWriter(f"{experiment_name}/{t}")
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

    def log_text(self, tag, text):
        self.writer.add_text(tag=tag, text_string=text)
