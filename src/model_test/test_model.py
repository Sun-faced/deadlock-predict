"""
model_tester.py - checks an already trained model on a test set, rank by rank.

A prediction counts as correct if:
    match outcome      it names the right winner
    time and kills     it's within 10% of the real value
    mid-bosses         rounded to a whole number, it equals the real count

It makes 6 plots, one per metric, with one stripe per rank. Each stripe is
100% of that rank's matches: orange = predicted correctly, blue = wrong, and
the shades show how long those matches lasted.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TOLERANCE = 0.10                     # "within 10%"
PREDICT_BATCH_SIZE = 4096
DURATION_EDGES = [20, 30, 40, 50]    # minutes, which split matches into the 5 buckets below
BUCKET_LABELS = ["under 20 min", "20-30 min", "30-40 min", "40-50 min", "50+ min"]
CORRECT_COLORS = ["#fd8f66", "#eb6834", "#cb4b0b", "#a33903", "#7a2a05"]  # orange, short -> long matches
WRONG_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]    # blue, short -> long matches
INK, SOFT_INK, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


class TestModel:
    def __init__(self, Model):
        self.Model = Model           # an already trained model object

    def test(self, test_inputs_path="data/ranked_test/inputs.npy", test_outputs_path="data/ranked_test/outputs.npy", out_dir="results", show=True):
        """Scores the model, prints a table, saves 6 plots to out_dir and returns
        {metric: {rank name: share correct}}. Ranks are read from rank_tiers.npy
        and info.json in the same folder as the inputs."""
        folder = Path(test_inputs_path).parent
        inputs = np.load(test_inputs_path, mmap_mode="r")
        outputs = np.load(test_outputs_path, mmap_mode="r")
        tiers = np.load(folder / "rank_tiers.npy", mmap_mode="r")
        rank_names = json.loads((folder / "info.json").read_text())["rank_names"]
        n = len(inputs)

        # 1. Check the six predictions in small batches so a large test set fits.
        correct = {}
        for start, end, batch in self._correct_batches(inputs, outputs, PREDICT_BATCH_SIZE):
            for metric, right in batch.items():
                if metric not in correct:
                    correct[metric] = np.empty(n, dtype=bool)
                correct[metric][start:end] = right
        rules = {
            "Match outcome": "Correct = picked the winning side",
            "Match time": "Correct = within 10% of the real match length",
            "Kills, side 0": "Correct = within 10% of the real kill count",
            "Kills, side 1": "Correct = within 10% of the real kill count",
            "Mid-bosses, side 0": "Correct = rounded prediction equals the real count",
            "Mid-bosses, side 1": "Correct = rounded prediction equals the real count",
        }

        # 2. Which length bucket each match falls into: 0 = under 20 min ... 4 = 50+ min.
        buckets = np.digitize(outputs[:, 1] / 60, DURATION_EDGES)

        # 3. One plot per metric. Matches with unknown rank (tier 0) are left out.
        ranks = [t for t in range(1, len(rank_names) + 1) if (tiers == t).any()]
        names = [rank_names[t - 1] for t in ranks]
        model_name = type(self.Model).__name__
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        results = {}
        for metric, right in correct.items():
            right_share, wrong_share = self._shares(right, buckets, tiers, ranks)
            results[metric] = dict(zip(names, right_share.sum(axis=1)))
            path = Path(out_dir) / f"{model_name}_{metric.replace(', ', '_').replace(' ', '_').lower()}.png"
            self._plot(metric, rules[metric], model_name, names, right_share, wrong_share,
                       [(tiers == t).sum() for t in ranks], right[tiers > 0].mean(), path)
            if not show:
                plt.close()

        self._print_table(results, names, correct, tiers)
        print(f"Plots saved in {Path(out_dir).resolve()}")
        if show:
            plt.show()
        return results

    def training_accuracy(self, inputs_path, outputs_path, save_path=None,
                          batch_size=PREDICT_BATCH_SIZE):
        """Score all training matches in batches, without making extra plots.
        The returned values are fractions between 0 and 1 for the six outputs."""
        inputs = np.load(inputs_path, mmap_mode="r")
        outputs = np.load(outputs_path, mmap_mode="r")
        if len(inputs) != len(outputs) or len(inputs) == 0:
            raise ValueError("inputs and outputs must have the same nonzero number of matches")

        totals = {}
        for _, _, correct in self._correct_batches(inputs, outputs, batch_size):
            for metric, right in correct.items():
                totals[metric] = totals.get(metric, 0) + int(right.sum())

        accuracy = {metric: total / len(inputs) for metric, total in totals.items()}
        print(f"\nTraining accuracy on {len(inputs):,} unranked matches (already seen by the model):")
        for metric, share in accuracy.items():
            print(f"  {metric:<22} {share:.1%}")

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"matches": len(inputs), "accuracy": accuracy}, indent=2))
            print(f"Training accuracy saved in {path.resolve()}")
        return accuracy

    def _correct_batches(self, inputs, outputs, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(inputs) != len(outputs) or len(inputs) == 0:
            raise ValueError("inputs and outputs must have the same nonzero number of matches")
        for start in range(0, len(inputs), batch_size):
            end = min(start + batch_size, len(inputs))
            yield start, end, self._correct(inputs[start:end], outputs[start:end])

    @staticmethod
    def _as_array(prediction, shape):
        """Model output -> decimal array of the given shape. A single number
        (or a single pair) from a model like the baseline is used for every match."""
        if hasattr(prediction, "detach"):        # a PyTorch tensor
            prediction = prediction.detach().cpu().numpy()
        return np.broadcast_to(np.asarray(prediction, dtype=float), shape)

    def _correct(self, inputs, outputs):
        """Right or wrong for each output of each match."""
        n = len(inputs)
        outcome = self._as_array(self.Model.predict_match_outcome(inputs), (n,))
        time = self._as_array(self.Model.predict_time(inputs), (n,))
        kills = self._as_array(self.Model.predict_kills(inputs), (n, 2))
        objectives = self._as_array(self.Model.predict_objective(inputs), (n, 2))
        return {
            "Match outcome": (outcome >= 0.5) == (outputs[:, 0] == 1),
            "Match time": self._within(time, outputs[:, 1]),
            "Kills, side 0": self._within(kills[:, 0], outputs[:, 2]),
            "Kills, side 1": self._within(kills[:, 1], outputs[:, 3]),
            "Mid-bosses, side 0": np.round(objectives[:, 0]) == outputs[:, 4],
            "Mid-bosses, side 1": np.round(objectives[:, 1]) == outputs[:, 5],
        }

    @staticmethod
    def _within(predicted, actual):
        return np.abs(predicted - actual) <= TOLERANCE * actual

    @staticmethod
    def _shares(right, buckets, tiers, ranks):
        """[ranks, 5] arrays: share of each rank's matches that were right (or wrong)
        and fell in each length bucket. Each rank's two rows add up to 1."""
        right_share = np.zeros((len(ranks), len(BUCKET_LABELS)))
        wrong_share = np.zeros((len(ranks), len(BUCKET_LABELS)))
        for i, t in enumerate(ranks):
            in_rank = tiers == t
            r, b = right[in_rank], buckets[in_rank]
            for k in range(len(BUCKET_LABELS)):
                right_share[i, k] = np.mean(r & (b == k))
                wrong_share[i, k] = np.mean(~r & (b == k))
        return right_share, wrong_share

    @staticmethod
    def _plot(metric, rule, model_name, names, right_share, wrong_share, counts, overall, path):
        fig, ax = plt.subplots(figsize=(13, 6.5))
        x = np.arange(len(names))
        bottom = np.zeros(len(names))

        # Stack: correct buckets first (orange), then wrong (blue) on top.
        for shares, colors, word in [(right_share, CORRECT_COLORS, "Correct"),
                                     (wrong_share, WRONG_COLORS, "Wrong")]:
            for k, label in enumerate(BUCKET_LABELS):
                ax.bar(x, shares[:, k] * 100, bottom=bottom, width=0.72, color=colors[k],
                       edgecolor="white", linewidth=1, label=f"{word}, {label}")
                bottom += shares[:, k] * 100

        # Share correct above each stripe.
        for i, share in enumerate(right_share.sum(axis=1)):
            ax.text(i, 102, f"{share:.1%}", ha="center", va="bottom", fontsize=10, color=INK)
        fig.suptitle(f"{metric}: {overall:.1%} correct overall ({model_name})",
                     x=0.06, ha="left", fontsize=14, color=INK)
        ax.set_title(f"{rule}. Each stripe is 100% of that rank's matches.",
                     loc="left", fontsize=10, color=SOFT_INK, pad=16)
        ax.set_xticks(x, [f"{name}\n{count:,}" for name, count in zip(names, counts)], color=SOFT_INK)
        ax.set_xlabel("Rank (number of test matches)", color=SOFT_INK)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], color=MUTED)
        ax.set_ylabel("Share of the rank's matches", color=SOFT_INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(length=0)

        # Legend in the same top-to-bottom order as the stack.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1),
                  frameon=False, fontsize=9, title="Prediction, match length",
                  title_fontsize=9, labelcolor=SOFT_INK)
        fig.tight_layout()
        fig.savefig(path, dpi=150)

    @staticmethod
    def _print_table(results, names, correct, tiers):
        short = ["outcome", "time", "kills 0", "kills 1", "midboss 0", "midboss 1"]
        print(f"\n{'rank':<11}" + "".join(f"{s:>11}" for s in short))
        for name in names:
            print(f"{name:<11}" + "".join(f"{results[m][name]:>11.1%}" for m in results))
        known = tiers > 0
        print(f"{'all':<11}" + "".join(f"{correct[m][known].mean():>11.1%}" for m in results))
