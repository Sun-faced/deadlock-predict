"""
training_data.py - read a dataset built by DatasetBuilder and hand it out in batches.

    from deadlock_data import TrainingData

    train = TrainingData("data/train", split="train")
    for inputs, outputs in train:        # one pass = one epoch
        ...                              # inputs: [batch, 63], outputs: [batch, 6]

    # Ranked test set: every match, or only one rank
    test = TrainingData("data/ranked_test", split="all")
    oracle = TrainingData("data/ranked_test", split="all", rank="Oracle")
    print(test.matches_per_rank())

Splits are by time, never random: "train" is the oldest 80% of matches,
"validation" the next 10%, "test" the newest 10%, "all" is everything.
Training batches come in a new random order every epoch; the others come in
time order.

The whole split is loaded once onto the GPU (or the CPU if there is no GPU), so
handing out batches is just slicing and costs almost nothing.
"""

import json
from pathlib import Path

import numpy as np
import torch

SPLITS = {"train": (0.0, 0.8), "validation": (0.8, 0.9), "test": (0.9, 1.0), "all": (0.0, 1.0)}


class TrainingData:
    def __init__(self, folder="data/train", split="train", batch_size=4096, shuffle=None,
                 device=None, rank=None):
        """rank: None for every match, or a rank name ("Oracle"), number (8), or a
        list of them, to keep only matches whose average rank is in that bucket."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}, not {split!r}")
        folder = Path(folder)
        info = json.loads((folder / "info.json").read_text())
        self.input_names = info["input_names"]    # what each of the 63 input columns is
        self.output_names = info["output_names"]  # what each of the 6 output columns is
        self.rank_names = info.get("rank_names", [])  # rank 1 = rank_names[0] ...
        self.split = split
        self.batch_size = batch_size
        self.shuffle = (split == "train") if shuffle is None else shuffle
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # mmap_mode="r" reads only the rows of this split from disk.
        inputs = np.load(folder / "inputs.npy", mmap_mode="r")
        lo, hi = (int(len(inputs) * f) for f in SPLITS[split])

        def part(file):
            path = folder / file
            return np.array(np.load(path, mmap_mode="r")[lo:hi]) if path.exists() else None

        self.rank_tiers = part("rank_tiers.npy")    # 1 = Initiate ... 11 = Eternus, 0 = unknown
        self.rank_scores = part("rank_scores.npy")  # average rank, 1-66 scale
        keep = np.ones(hi - lo, dtype=bool)
        if rank is not None:
            if self.rank_tiers is None:
                raise ValueError("This dataset has no ranks. Rebuild it with the current DatasetBuilder.")
            wanted = [self._rank_number(r) for r in (rank if isinstance(rank, (list, tuple)) else [rank])]
            keep = np.isin(self.rank_tiers, wanted)
            self.rank_tiers, self.rank_scores = self.rank_tiers[keep], self.rank_scores[keep]

        self.match_ids = part("match_ids.npy")[keep]
        self.start_times = part("start_times.npy")[keep]
        # Kept as 16-bit integers to save memory; each batch is widened when handed out.
        self.inputs = torch.from_numpy(np.array(inputs[lo:hi])[keep]).to(self.device)
        outputs = np.load(folder / "outputs.npy", mmap_mode="r")
        self.outputs = torch.from_numpy(np.array(outputs[lo:hi])[keep]).to(self.device)

    def __len__(self):
        """Number of batches in one epoch."""
        return -(-len(self.inputs) // self.batch_size)

    def __iter__(self):
        n = len(self.inputs)
        order = torch.randperm(n, device=self.device) if self.shuffle else None
        for i in range(0, n, self.batch_size):
            rows = order[i:i + self.batch_size] if order is not None else slice(i, i + self.batch_size)
            yield self.inputs[rows].long(), self.outputs[rows].long()

    @property
    def num_matches(self):
        return len(self.inputs)

    def column(self, column_name):
        """All values of one input or output column, e.g. data.column("duration_s")."""
        if column_name in self.input_names:
            return self.inputs[:, self.input_names.index(column_name)].long()
        return self.outputs[:, self.output_names.index(column_name)].long()

    def matches_per_rank(self):
        """{"Initiate": 1234, ...} for the matches in this split."""
        if self.rank_tiers is None:
            return {}
        names = ["unknown"] + self.rank_names
        counts = np.bincount(self.rank_tiers, minlength=len(names))
        return {names[i]: int(c) for i, c in enumerate(counts) if c}

    def _rank_number(self, rank):
        if isinstance(rank, (int, np.integer)):
            return int(rank)
        lowered = [r.lower() for r in self.rank_names]
        if rank.lower() not in lowered:
            raise ValueError(f"Unknown rank {rank!r}. Ranks: {', '.join(self.rank_names)}")
        return lowered.index(rank.lower()) + 1

    def __repr__(self):
        return (f"TrainingData(split={self.split!r}, matches={self.num_matches:,}, "
                f"batches={len(self)} of {self.batch_size}, device={self.device})")
