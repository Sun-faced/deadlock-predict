"""
This is a linear model. Every hero gets 6 learned numbers, and a prediction is
a starting value plus the sum of those numbers for the heroes in the match:
1. Win:   side bias + strengths of side 0 - strengths of side 1, squashed to a chance
2. Time:  average time + time effects of all 12 heroes
3. Kills of a side:      average + "kills made" of its heroes + "kills given" of the enemy heroes
4. Mid-bosses of a side: average + "mid-bosses taken" of its heroes + "mid-bosses given" of the enemy heroes
It starts out exactly like the baseline (every hero number 0) and learns from there.
"""

import numpy as np
import torch
import torch.nn as nn

from data import TrainingData


class LinearModel(nn.Module):
    def __init__(self, folder="data/train"):
        super().__init__()
        self.folder = folder
        # One learned number per hero id for each effect. All start at 0.
        self.strength = self._hero_table()
        self.time_effect = self._hero_table()     # minutes
        self.kills_made = self._hero_table()
        self.kills_given = self._hero_table()
        self.bosses_taken = self._hero_table()
        self.bosses_given = self._hero_table()
        # Starting values: [win score, minutes, kills side 0, kills side 1,
        # mid-bosses side 0, mid-bosses side 1]. Set to the averages in fit().
        self.start = nn.Parameter(torch.zeros(6))

    @staticmethod
    def _hero_table():
        table = nn.Embedding(256, 1)
        nn.init.zeros_(table.weight)
        return table

    def forward(self, heroes):
        """heroes: [matches, 12] hero ids. Returns win score, minutes, kills [matches, 2]
        and mid-bosses [matches, 2]. A win score above 0 favours side 0."""
        side0, side1 = heroes[:, :6], heroes[:, 6:]

        def total(table, team):              # add up one number over the heroes of a team
            return table(team).sum(dim=(1, 2))

        win_score = self.start[0] + total(self.strength, side0) - total(self.strength, side1)
        minutes = self.start[1] + total(self.time_effect, heroes)
        kills = torch.stack([total(self.kills_made, side0) + total(self.kills_given, side1),
                             total(self.kills_made, side1) + total(self.kills_given, side0)], dim=1)
        bosses = torch.stack([total(self.bosses_taken, side0) + total(self.bosses_given, side1),
                              total(self.bosses_taken, side1) + total(self.bosses_given, side0)], dim=1)
        return win_score, minutes, kills + self.start[2:4], bosses + self.start[4:6]

    def fit(self, epochs=10):
        """Trains on the "train" split, the oldest 80% of the matches.
        (Called fit, not train: PyTorch models already have a train() method.)"""
        data = TrainingData(self.folder, split="train", batch_size=8192)
        self.to(data.device)

        # Start as the baseline: hero numbers 0, starting values = training averages.
        outputs = data.outputs.float()
        with torch.no_grad():
            side0_rate = outputs[:, 0].mean()
            self.start[0] = torch.log(side0_rate / (1 - side0_rate))
            self.start[1] = outputs[:, 1].mean() / 60
            self.start[2:] = outputs[:, 2:].mean(dim=0)

        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01, weight_decay=1e-4)
        for epoch in range(1, epochs + 1):
            for inputs, outputs in data:
                loss = self._loss(inputs[:, :12], outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"epoch {epoch}/{epochs} done")

    def _loss(self, heroes, outputs):
        """Win loss + time loss + kills loss + mid-bosses loss for one batch.
        Output columns: 0 team0_won, 1 duration_s, 2-3 kills, 4-5 mid-bosses."""
        win_score, minutes, kills, bosses = self(heroes)
        outputs = outputs.float()
        return (nn.functional.binary_cross_entropy_with_logits(win_score, outputs[:, 0])
                + nn.functional.mse_loss(minutes, outputs[:, 1] / 60)
                + nn.functional.mse_loss(kills, outputs[:, 2:4])
                + nn.functional.mse_loss(bosses, outputs[:, 4:6]))

    @torch.no_grad()
    def _predict(self, inputs):
        """The 4 raw predictions for one match (63 numbers) or many ([matches, 63])."""
        if self.start[1] == 0:   # fit() sets this to the average match length
            raise RuntimeError("LinearModel isn't trained. Call model.fit() first "
                               "(model.train() is PyTorch's mode switch and doesn't train).")
        heroes = np.atleast_2d(np.asarray(inputs))[:, :12]
        heroes = torch.as_tensor(heroes, dtype=torch.long, device=self.start.device)
        return self(heroes)

    def predict_match_outcome(self, inputs):
        """Chance that side 0 wins, per match. 0.5 or more means side 0 is predicted to win."""
        return torch.sigmoid(self._predict(inputs)[0]).cpu().numpy()

    def predict_time(self, inputs):
        """Match length in seconds, per match."""
        return (self._predict(inputs)[1] * 60).cpu().numpy()

    def predict_kills(self, inputs):
        """[kills of side 0, kills of side 1], per match."""
        return self._predict(inputs)[2].cpu().numpy()

    def predict_objective(self, inputs):
        """[mid-bosses of side 0, mid-bosses of side 1], per match."""
        return self._predict(inputs)[3].cpu().numpy()