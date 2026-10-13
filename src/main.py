"""Example main: make sure both datasets exist, then load them.

The first run builds what's missing (asking Y/N before downloading). Later runs
find the datasets already on disk and go straight to loading them.
"""

from data import TrainingData, ensure_dataset
from models.basic import Baseline
from models.linear import LinearModel
from model_test.test_model import TestModel

if __name__ == "__main__":
    """
    Check that the data is here
    """
    train_folder = ensure_dataset("data/train", start="2026-08-01", end="2026-09-29")
    test_folder = ensure_dataset("data/ranked_test", start="2026-08-01", end="2026-09-29", match_mode="Ranked")
    if train_folder is None or test_folder is None:
        raise SystemExit("No dataset, stopping.")

    """
    Start the training of baseline model
    """
    model_basic = Baseline(train_folder)
    model_basic.train()
    TestModel(model_basic).test(test_folder / "inputs.npy", test_folder / "outputs.npy")

    """
    Start the testing of linear model
    """
    model_linear = LinearModel(train_folder)
    model_linear.fit()
    TestModel(model_linear).test(test_folder / "inputs.npy", test_folder / "outputs.npy")


