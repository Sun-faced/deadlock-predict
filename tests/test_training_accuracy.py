import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import TrainingData
from model_test.test_model import TestModel
from models.linear import LinearModel
from models.mlp import TaskMLP


class ConstantModel:
    def __init__(self):
        self.batch_sizes = []

    def predict_match_outcome(self, inputs):
        self.batch_sizes.append(len(inputs))
        return np.full(len(inputs), 0.75)

    def predict_time(self, inputs):
        return np.full(len(inputs), 100)

    def predict_kills(self, inputs):
        return np.full((len(inputs), 2), 10)

    def predict_objective(self, inputs):
        return np.full((len(inputs), 2), 1)


class TrainingAccuracyTests(unittest.TestCase):
    def test_scores_every_training_match_in_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            inputs = np.zeros((4, 63), dtype=np.int16)
            outputs = np.array([
                [1, 100, 10, 10, 1, 1],
                [0, 200, 20, 20, 0, 0],
                [1, 100, 10, 10, 1, 1],
                [0, 200, 20, 20, 0, 0],
            ], dtype=np.int16)
            np.save(folder / "inputs.npy", inputs)
            np.save(folder / "outputs.npy", outputs)
            model = ConstantModel()
            save_path = folder / "training_accuracy.json"

            accuracy = TestModel(model).training_accuracy(
                folder / "inputs.npy", folder / "outputs.npy", save_path=save_path,
                batch_size=2)

            self.assertEqual(model.batch_sizes, [2, 2])
            self.assertEqual(len(accuracy), 6)
            self.assertTrue(all(share == 0.5 for share in accuracy.values()))
            saved = json.loads(save_path.read_text())
            self.assertEqual(saved["matches"], 4)
            self.assertEqual(saved["accuracy"], accuracy)

            np.save(folder / "rank_tiers.npy", np.ones(4, dtype=np.int8))
            (folder / "info.json").write_text(json.dumps({"rank_names": ["Initiate"]}))
            with patch("model_test.test_model.PREDICT_BATCH_SIZE", 2):
                ranked = TestModel(model).test(
                    folder / "inputs.npy", folder / "outputs.npy",
                    out_dir=folder / "plots", show=False)
            self.assertEqual(model.batch_sizes, [2, 2, 2, 2])
            self.assertTrue(all(scores["Initiate"] == 0.5 for scores in ranked.values()))

    def test_models_train_on_all_matches_when_validation_is_off(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            inputs = np.zeros((10, 63), dtype=np.int16)
            inputs[:, 24:36] = 1  # valid party sizes
            outputs = np.tile([0, 1200, 10, 10, 1, 1], (10, 1)).astype(np.int16)
            outputs[-2:, 0] = 1
            np.save(folder / "inputs.npy", inputs)
            np.save(folder / "outputs.npy", outputs)
            for name in ("rank_tiers", "rank_scores", "match_ids", "start_times"):
                np.save(folder / f"{name}.npy", np.zeros(10))
            (folder / "info.json").write_text(json.dumps({
                "input_names": [str(i) for i in range(63)],
                "output_names": [str(i) for i in range(6)],
            }))

            with patch("models.linear.TrainingData", wraps=TrainingData) as linear_data:
                LinearModel(folder).fit(epochs=1, patience=-1)
            self.assertEqual(linear_data.call_args.kwargs["split"], "all")
            self.assertTrue(linear_data.call_args.kwargs["shuffle"])

            with patch("models.mlp.TrainingData", wraps=TrainingData) as mlp_data:
                TaskMLP("win", folder).fit(epochs=1, patience=-1)
            self.assertEqual(mlp_data.call_args.kwargs["split"], "all")
            self.assertTrue(mlp_data.call_args.kwargs["shuffle"])


if __name__ == "__main__":
    unittest.main()
