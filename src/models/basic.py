"""
This is a baseline class which is supposed not to use any kind of
ML. The ways it predicts is simple:
1. It predicts that the side which won more often in training wins every match.
2. It predicts the avg time of match
3. It predicts the avg kills for each side
4. It predicts the avg number of objectives (mid-bosses) for each side.
"""

import numpy as np
from pathlib import Path

from .layout import BOSSES_OUTPUTS, KILLS_OUTPUTS, TIME_OUTPUT, WIN_OUTPUT


MAJORITY_WIN_RATE = 0.5


class Baseline:
    def __init__(self, folder="data/train"):
        self.inputs = np.load(Path(folder) / "inputs.npy")
        self.outputs = np.load(Path(folder) / "outputs.npy")
        self.winner = None                   # 1 if side 0 win  s more often, 0 if side 1
        self.avg_time = None                 # seconds
        self.avg_kills = None                # [side 0, side 1]
        self.avg_objectives = None           # [side 0, side 1]

    def train(self):
        side0_win_rate = self.outputs[:, WIN_OUTPUT].mean()
        self.winner = 1 if side0_win_rate > MAJORITY_WIN_RATE else 0
        self.avg_time = self.outputs[:, TIME_OUTPUT].mean()
        self.avg_kills = self.outputs[:, KILLS_OUTPUTS].mean(axis=0).tolist()
        self.avg_objectives = self.outputs[:, BOSSES_OUTPUTS].mean(axis=0).tolist()

    def predict_match_outcome(self, inputs=None):
        """1 if side 0 is predicted to win, 0 if side 1.
        The baseline ignores the match and always picks the side that wins more often."""
        return self.winner

    def predict_time(self, inputs=None):
        """Match length in seconds. The baseline ignores the match and returns the average."""
        return self.avg_time

    def predict_kills(self, inputs=None):
        """[kills of side 0, kills of side 1]."""
        return self.avg_kills

    def predict_objective(self, inputs=None):
        """[mid-bosses of side 0, mid-bosses of side 1]."""
        return self.avg_objectives
