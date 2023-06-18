from typing import Tuple

from pathlib import Path

import pandas as pd

from tqdm import tqdm
from slugify import slugify
from tensorboard.backend.event_processing import event_accumulator

from loguru import logger


EXPERIMENTS_ROOT_PATH = Path("runs")


def get_event_data(event_path: str):
    event_path = str(event_path)
    ea = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        },
    )

    ea.Reload()

    return ea.Scalars("f1_macro")


class ExperimentSerializer:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.path = EXPERIMENTS_ROOT_PATH / self.experiment_name

    def get_runs(self) -> Tuple:
        paths = list(self.path.iterdir())
        return paths

    def get_f1_macro_test_records(self, run_path: Path):
        metric_folder = run_path / "f1_macro_test"

        event_path = list(metric_folder.iterdir())[0]
        event_data = get_event_data(event_path)

        f1_macro_values = [(scalar.step, scalar.value) for scalar in event_data]

        return pd.DataFrame(f1_macro_values, columns=["step", "f1_macro_test"])

    def serialize(self):
        logger.info(f"Serializing experiment: {self.experiment_name}...")
        runs = self.get_runs()

        runs_dfs = []

        for run in tqdm(runs):
            run_df = self.get_f1_macro_test_records(run_path=run)
            run_df["experiment_name"] = self.experiment_name
            run_df["run"] = run.name
            runs_dfs.append(run_df)
        
        return pd.concat(runs_dfs)


def main():
    experiments = [
        "XRAY-64x64-SYNTHETIC-UNBALANCED",
        "XRAY-64x64-REAL"
    ]

    for exp_name in experiments:
        exp = ExperimentSerializer(experiment_name=exp_name)
        df = exp.serialize()
        df.to_parquet(f"data/results/{slugify(exp_name, separator='_')}.parquet", index=False)

    logger.info("Experiments serialization runs successfully.")


if __name__ == "__main__":
    main()