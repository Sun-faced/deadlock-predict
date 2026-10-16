"""
This is a linear model. By default, it learns from hero IDs only. Other groups
of match inputs can be turned on when creating the model. Each prediction is
a starting value plus the effects of the selected inputs:
1. Win: side bias + effects of side 0 - effects of side 1, squashed to a chance
2. Time: average time + effects of both sides
3. Kills of a side: average + "kills made" effects of its players + "kills given" effects of the enemy
4. Mid-bosses of a side: average + "bosses taken" effects of its players + "bosses given" effects of the enemy
It starts out exactly like the baseline (every feature weight 0) and learns from there
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from data import TrainingData
from .layout import (BOSSES_OUTPUTS, EXPERIENCE_MATCH_SCALE, EXPERIENCE_VALUE_COUNT,
                     FLAG_COUNT, FLAG_INPUTS, GAMES_INPUTS, HERO_ID_COUNT, HERO_INPUTS,
                     KILLS_OUTPUTS, LANE_CODES, LANE_INPUTS, OUTPUT_COUNT,
                     PARTY_INPUTS, PARTY_SIZE_COUNT, SECONDS_PER_MINUTE, SIDE0_PLAYERS,
                     SIDE1_PLAYERS, TIME_OUTPUT, WIN_OUTPUT, WINS_INPUTS)


TRAIN_BATCH_SIZE = 8192
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4


class LinearModel(nn.Module):
    def __init__(self, folder="data/train", features=None):
        super().__init__()
        self.folder = folder
        self.features = {
            "ID": True,
            "LANES": False,
            "PARTIES": False,
            "EXPERIENCE": False,
            "FLAGS": False,
        }
        if features is not None:
            unknown = set(features) - set(self.features)
            if unknown:
                raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
            for name, enabled in features.items():
                if not isinstance(enabled, bool):
                    raise TypeError(f"{name} must be True or False")
            self.features.update(features)

        # Each selected player group adds positions to the team vector.
        size = 0
        if self.features["ID"]:
            size += HERO_ID_COUNT
        if self.features["LANES"]:
            size += len(LANE_CODES)
        if self.features["PARTIES"]:
            size += PARTY_SIZE_COUNT
        if self.features["EXPERIENCE"]:
            size += EXPERIENCE_VALUE_COUNT

        # Each prediction uses its own vector of learned player weights.
        self.strength = nn.Parameter(torch.zeros(size))
        self.time_effect = nn.Parameter(torch.zeros(size))  # minutes
        self.kills_made = nn.Parameter(torch.zeros(size))
        self.kills_given = nn.Parameter(torch.zeros(size))
        self.bosses_taken = nn.Parameter(torch.zeros(size))
        self.bosses_given = nn.Parameter(torch.zeros(size))
        if self.features["FLAGS"]:
            # One flag weight for each output number.
            self.flag_weights = nn.Parameter(torch.zeros(OUTPUT_COUNT, FLAG_COUNT))

        # Starting values: [win score, minutes, kills side 0, kills side 1,
        # mid-bosses side 0, mid-bosses side 1]. Set to the averages in fit().
        self.start = nn.Parameter(torch.zeros(OUTPUT_COUNT))

    def forward(self, inputs):
        """Return win score, minutes, kills and mid-bosses for each match.
        A win score above 0 favours side 0."""
        side0_parts = []
        side1_parts = []

        if self.features["ID"]:
            # Each hero ID becomes a one-hot vector with a 1 at that ID.
            heroes = inputs[:, HERO_INPUTS]
            side0_heroes = nn.functional.one_hot(heroes[:, SIDE0_PLAYERS].long(), HERO_ID_COUNT)
            side1_heroes = nn.functional.one_hot(heroes[:, SIDE1_PLAYERS].long(), HERO_ID_COUNT)
            side0_parts.append(side0_heroes.sum(dim=1).float())
            side1_parts.append(side1_heroes.sum(dim=1).float())

        if self.features["LANES"]:
            # Count the known lane codes. Unknown codes count as 0.
            lanes = inputs[:, LANE_INPUTS]
            lanes0 = lanes[:, SIDE0_PLAYERS]
            lanes1 = lanes[:, SIDE1_PLAYERS]
            side0_counts = [(lanes0 == lane).sum(dim=1) for lane in LANE_CODES]
            side1_counts = [(lanes1 == lane).sum(dim=1) for lane in LANE_CODES]
            side0_parts.append(torch.stack(side0_counts, dim=1).float())
            side1_parts.append(torch.stack(side1_counts, dim=1).float())

        if self.features["PARTIES"]:
            # Count solo players, duo players, and so on for each side.
            parties = inputs[:, PARTY_INPUTS]
            side0_parties = nn.functional.one_hot(parties[:, SIDE0_PLAYERS].long(), PARTY_SIZE_COUNT)
            side1_parties = nn.functional.one_hot(parties[:, SIDE1_PLAYERS].long(), PARTY_SIZE_COUNT)
            side0_parts.append(side0_parties.sum(dim=1).float())
            side1_parts.append(side1_parties.sum(dim=1).float())

        if self.features["EXPERIENCE"]:
            # Keep the two stored counts: games played and games won on this hero.
            # Use units of 100 matches so these numbers stay near the other inputs.
            games = inputs[:, GAMES_INPUTS].float() / EXPERIENCE_MATCH_SCALE
            wins = inputs[:, WINS_INPUTS].float() / EXPERIENCE_MATCH_SCALE
            side0_parts.append(torch.stack([games[:, SIDE0_PLAYERS].sum(dim=1),
                                            wins[:, SIDE0_PLAYERS].sum(dim=1)], dim=1))
            side1_parts.append(torch.stack([games[:, SIDE1_PLAYERS].sum(dim=1),
                                            wins[:, SIDE1_PLAYERS].sum(dim=1)], dim=1))

        if side0_parts:
            side0 = torch.cat(side0_parts, dim=1)
            side1 = torch.cat(side1_parts, dim=1)
        else:
            # This also allows a model that uses only the match-wide flags.
            side0 = torch.zeros((len(inputs), 0), device=inputs.device)
            side1 = torch.zeros((len(inputs), 0), device=inputs.device)

        if self.features["FLAGS"]:
            flag_effects = inputs[:, FLAG_INPUTS].float() @ self.flag_weights.T
        else:
            flag_effects = torch.zeros((len(inputs), OUTPUT_COUNT), device=inputs.device)

        # Team difference predicts the winner; team sum predicts match length.
        win_score = self.start[WIN_OUTPUT] + (side0 - side1) @ self.strength + flag_effects[:, WIN_OUTPUT]
        minutes = self.start[TIME_OUTPUT] + (side0 + side1) @ self.time_effect + flag_effects[:, TIME_OUTPUT]
        kills = torch.stack([side0 @ self.kills_made + side1 @ self.kills_given,
                             side1 @ self.kills_made + side0 @ self.kills_given], dim=1)
        bosses = torch.stack([side0 @ self.bosses_taken + side1 @ self.bosses_given,
                              side1 @ self.bosses_taken + side0 @ self.bosses_given], dim=1)
        return (win_score, minutes,
                kills + self.start[KILLS_OUTPUTS] + flag_effects[:, KILLS_OUTPUTS],
                bosses + self.start[BOSSES_OUTPUTS] + flag_effects[:, BOSSES_OUTPUTS])

    def fit(self, epochs=10, patience=-1):
        """Train on every match when patience is -1. A positive patience keeps
        the time-based training and validation split for early stopping."""
        if patience != -1 and patience < 1:
            raise ValueError("patience must be -1 or a positive number of epochs")
        training_split = "all" if patience == -1 else "train"
        data = TrainingData(self.folder, split=training_split, batch_size=TRAIN_BATCH_SIZE,
                            shuffle=True)
        validation = None

        if patience != -1:
            validation = TrainingData(self.folder, split="validation", batch_size=TRAIN_BATCH_SIZE)
            if validation.num_matches == 0:
                raise ValueError("The validation split has no matches")

        if data.device.type != "cuda":
            print(f"!=========================ATTENTION=========================!")
            print(f"LinearModel is training on {data.device}; CUDA is not in use.")
            print(f"=============================================================")
        self.to(data.device)

        # Start as the baseline: feature weights 0, starting values = training averages.
        outputs = data.outputs.float()
        with torch.no_grad():
            side0_rate = outputs[:, WIN_OUTPUT].mean()
            self.start[WIN_OUTPUT] = torch.log(side0_rate / (1 - side0_rate))
            self.start[TIME_OUTPUT] = outputs[:, TIME_OUTPUT].mean() / SECONDS_PER_MINUTE
            self.start[KILLS_OUTPUTS.start:] = outputs[:, KILLS_OUTPUTS.start:].mean(dim=0)

        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_loss = float("inf")
        best_epoch = 0
        best_weights = None
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            self.train()
            for inputs, outputs in data:
                loss = self._loss(inputs, outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if validation is None:
                print(f"epoch {epoch}/{epochs} done")
                continue

            self.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for inputs, outputs in validation:
                    validation_loss += self._loss(inputs, outputs).item() * len(inputs)
            validation_loss /= validation.num_matches
            if not np.isfinite(validation_loss):
                raise RuntimeError(f"Validation loss is not finite at epoch {epoch}")
            print(f"epoch {epoch}/{epochs}: validation loss {validation_loss:.4f}")

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Stopping after {patience} epochs without validation improvement.")
                    break

        if best_weights is not None:
            self.load_state_dict(best_weights)
            print(f"Using epoch {best_epoch} (validation loss {best_loss:.4f}).")

    def _loss(self, inputs, outputs):
        """Win loss + time loss + kills loss + mid-bosses loss for one batch.
        Output columns are named in layout.py."""
        win_score, minutes, kills, bosses = self(inputs)
        outputs = outputs.float()
        return (nn.functional.binary_cross_entropy_with_logits(win_score, outputs[:, WIN_OUTPUT])
                + nn.functional.mse_loss(minutes, outputs[:, TIME_OUTPUT] / SECONDS_PER_MINUTE)
                + nn.functional.mse_loss(kills, outputs[:, KILLS_OUTPUTS])
                + nn.functional.mse_loss(bosses, outputs[:, BOSSES_OUTPUTS]))

    @torch.no_grad()
    def _predict(self, inputs):
        """The 4 raw predictions for one match or many matches."""
        if self.start[TIME_OUTPUT] == 0:   # fit() sets this to the average match length
            raise RuntimeError("LinearModel isn't trained. Call model.fit() first "
                               "(model.train() is PyTorch's mode switch and doesn't train).")
        inputs = np.atleast_2d(np.asarray(inputs))
        inputs = torch.as_tensor(inputs, dtype=torch.long, device=self.start.device)
        return self(inputs)

    def predict_match_outcome(self, inputs):
        """Chance that side 0 wins, per match. 0.5 or more means side 0 is predicted to win."""
        return torch.sigmoid(self._predict(inputs)[0]).cpu().numpy()

    def predict_time(self, inputs):
        """Match length in seconds, per match."""
        return (self._predict(inputs)[1] * SECONDS_PER_MINUTE).cpu().numpy()

    def predict_kills(self, inputs):
        """[kills of side 0, kills of side 1], per match."""
        return self._predict(inputs)[2].cpu().numpy()

    def predict_objective(self, inputs):
        """[mid-bosses of side 0, mid-bosses of side 1], per match."""
        return self._predict(inputs)[3].cpu().numpy()
