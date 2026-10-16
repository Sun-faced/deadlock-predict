"""Independent MLPs for win, time, kills, and mid-boss predictions."""

import copy

import numpy as np
import torch
import torch.nn as nn

from data import TrainingData
from .layout import (BOSSES_OUTPUTS, EXPERIENCE_MATCH_SCALE, EXPERIENCE_VALUE_COUNT,
                     FLAG_COUNT, FLAG_INPUTS, GAMES_INPUTS, HERO_ID_COUNT, HERO_INPUTS,
                     INPUT_COUNT, KILLS_OUTPUTS, LANE_CODES, LANE_INPUTS,
                     LANE_POSITION_COUNT, PARTY_INPUTS, PARTY_SIZE_COUNT,
                     SECONDS_PER_MINUTE, SIDE0_PLAYERS, SIDE1_PLAYERS,
                     TEAMS_PER_MATCH, TIME_OUTPUT, WIN_OUTPUT, WINS_INPUTS)


HERO_EMBEDDING_SIZE = 16
PLAYER_HIDDEN_SIZE = 32
MATCHUP_HIDDEN_SIZE = 64
PLAYER_INPUT_SIZE = (HERO_EMBEDDING_SIZE + LANE_POSITION_COUNT + PARTY_SIZE_COUNT
                     + EXPERIENCE_VALUE_COUNT)
MATCHUP_INPUT_SIZE = TEAMS_PER_MATCH * PLAYER_HIDDEN_SIZE + FLAG_COUNT
TRAIN_BATCH_SIZE = 4096
PREDICT_BATCH_SIZE = TRAIN_BATCH_SIZE
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4


class TaskMLP(nn.Module):
    """One complete network trained for one of the four prediction tasks."""

    def __init__(self, task, folder="data/train"):
        super().__init__()
        if task not in ("win", "time", "kills", "bosses"):
            raise ValueError("task must be win, time, kills, or bosses")
        self.task = task
        self.folder = folder

        # A hero ID looks up a learned vector for this task
        self.hero_embedding = nn.Embedding(HERO_ID_COUNT, HERO_EMBEDDING_SIZE)

        # Per player: hero vector, lane category, party size, games and wins
        self.player_network = nn.Sequential(
            nn.Linear(PLAYER_INPUT_SIZE, PLAYER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(PLAYER_HIDDEN_SIZE, PLAYER_HIDDEN_SIZE),
            nn.ReLU(),
        )

        # Each matchup sees both team vectors and the flags
        self.matchup_network = nn.Sequential(
            nn.Linear(MATCHUP_INPUT_SIZE, MATCHUP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(MATCHUP_HIDDEN_SIZE, PLAYER_HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.head = nn.Linear(PLAYER_HIDDEN_SIZE, 1, bias=(task != "win"))
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        self.register_buffer("_trained", torch.tensor(False))

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != INPUT_COUNT:
            raise ValueError(f"inputs must have shape [matches, {INPUT_COUNT}]")

        heroes = inputs[:, HERO_INPUTS].long()
        lanes = inputs[:, LANE_INPUTS]
        parties = inputs[:, PARTY_INPUTS].long()
        games = inputs[:, GAMES_INPUTS].float() / EXPERIENCE_MATCH_SCALE
        wins = inputs[:, WINS_INPUTS].float() / EXPERIENCE_MATCH_SCALE
        flags = inputs[:, FLAG_INPUTS].float()

        # Treat lane codes outside LANE_CODES as unknown
        lane_positions = torch.zeros_like(lanes, dtype=torch.long)
        for position, code in enumerate(LANE_CODES, start=1):
            lane_positions[lanes == code] = position

        player_inputs = torch.cat((
            self.hero_embedding(heroes),
            nn.functional.one_hot(lane_positions, LANE_POSITION_COUNT).float(),
            nn.functional.one_hot(parties, PARTY_SIZE_COUNT).float(),
            torch.stack((games, wins), dim=-1),
        ), dim=-1)
        players = self.player_network(player_inputs)
        team0 = players[:, SIDE0_PLAYERS].sum(dim=1)
        team1 = players[:, SIDE1_PLAYERS].sum(dim=1)

        # Use the same matchup layers with the teams
        side0 = self.matchup_network(torch.cat((team0, team1, flags), dim=1))
        side1 = self.matchup_network(torch.cat((team1, team0, flags), dim=1))

        if self.task == "win":
            return self.head(side0 - side1).squeeze(1)
        if self.task == "time":
            return self.head(side0 + side1).squeeze(1)

        # Kills and mid-bosses each have one answer for each side.
        return torch.cat((self.head(side0), self.head(side1)), dim=1)

    def fit(self, epochs=30, patience=-1):
        """Train on every match when patience is -1. A positive patience keeps
        the time-based training and validation split for early stopping."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if patience != -1 and patience < 1:
            raise ValueError("patience must be -1 or a positive number of epochs")

        training_split = "all" if patience == -1 else "train"
        data = TrainingData(self.folder, split=training_split, batch_size=TRAIN_BATCH_SIZE,
                            shuffle=True)
        if data.num_matches == 0:
            raise ValueError("The training split has no matches")
        validation = None
        if patience != -1:
            validation = TrainingData(self.folder, split="validation", batch_size=TRAIN_BATCH_SIZE)
            if validation.num_matches == 0:
                raise ValueError("The validation split has no matches")

        if data.device.type != "cuda":
            print(f"============================ATTENTION============================")
            print(f"{self.task} MLP is training on {data.device}; CUDA is not in use.")
            print(f"============================ATTENTION============================")
        self.to(data.device)
        self._trained.fill_(False)

        # Start each regression task at the average of its training targets.
        outputs = data.outputs.float()
        with torch.no_grad():
            if self.task == "time":
                self.head.bias.fill_(outputs[:, TIME_OUTPUT].mean() / SECONDS_PER_MINUTE)
            if self.task == "kills":
                self.head.bias.fill_(outputs[:, KILLS_OUTPUTS].mean())
            if self.task == "bosses":
                self.head.bias.fill_(outputs[:, BOSSES_OUTPUTS].mean())

        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_loss = float("inf")
        best_epoch = 0
        best_weights = None
        epochs_without_improvement = 0

        print(f"Training MLP for {self.task}...")
        for epoch in range(1, epochs + 1):
            self.train()
            for inputs, batch_outputs in data:
                loss = self._loss(self(inputs), batch_outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if validation is None:
                print(f"{self.task} epoch {epoch}/{epochs} done")
                continue

            self.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for inputs, batch_outputs in validation:
                    loss = self._loss(self(inputs), batch_outputs)
                    validation_loss += loss.item() * len(inputs)
            validation_loss /= validation.num_matches
            if not np.isfinite(validation_loss):
                raise RuntimeError(f"{self.task} validation loss is not finite at epoch {epoch}")
            print(f"{self.task} epoch {epoch}/{epochs}: validation loss {validation_loss:.4f}")

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Stopping {self.task} after {patience} epochs without improvement.")
                    break

        if best_weights is not None:
            self.load_state_dict(best_weights)
            print(f"Using {self.task} epoch {best_epoch} (validation loss {best_loss:.4f}).")
        self._trained.fill_(True)

    def _loss(self, prediction, outputs):
        """Compare this task's predictions"""
        outputs = outputs.float()
        if self.task == "win":
            return nn.functional.binary_cross_entropy_with_logits(prediction, outputs[:, WIN_OUTPUT])
        if self.task == "time":
            return nn.functional.mse_loss(prediction, outputs[:, TIME_OUTPUT] / SECONDS_PER_MINUTE)
        if self.task == "kills":
            return nn.functional.mse_loss(prediction, outputs[:, KILLS_OUTPUTS])
        return nn.functional.mse_loss(prediction, outputs[:, BOSSES_OUTPUTS])

    @torch.no_grad()
    def predict(self, inputs):
        """Return this task's raw predictions, one test batch at a time."""
        if not self._trained.item():
            raise RuntimeError(f"{self.task} MLP isn't trained. Call fit() first.")
        rows = np.atleast_2d(np.asarray(inputs))
        if rows.shape[1] != INPUT_COUNT:
            raise ValueError(f"inputs must have shape [matches, {INPUT_COUNT}]")
        self.eval()

        shape = (len(rows), 2) if self.task in ("kills", "bosses") else (len(rows),)
        predictions = np.empty(shape, dtype=np.float32)
        for start in range(0, len(rows), PREDICT_BATCH_SIZE):
            end = min(start + PREDICT_BATCH_SIZE, len(rows))
            batch = torch.as_tensor(rows[start:end], dtype=torch.long,
                                    device=self.hero_embedding.weight.device)
            predictions[start:end] = self(batch).cpu().numpy()
        return predictions


class MLPModel(nn.Module):
    """Combine independently trained tasks for the usual tests"""

    def __init__(self, folder="data/train", win_model=None, time_model=None,
                 kills_model=None, bosses_model=None):
        super().__init__()
        self.win_model = win_model if win_model is not None else TaskMLP("win", folder)
        self.time_model = time_model if time_model is not None else TaskMLP("time", folder)
        self.kills_model = kills_model if kills_model is not None else TaskMLP("kills", folder)
        self.bosses_model = bosses_model if bosses_model is not None else TaskMLP("bosses", folder)

    def forward(self, inputs):
        """Return win logit, minutes, kills [2], and mid-bosses [2]."""
        return (self.win_model(inputs), self.time_model(inputs),
                self.kills_model(inputs), self.bosses_model(inputs))

    def fit(self, epochs=30, patience=-1):
        """Convenience method for training all four tasks."""
        self.win_model.fit(epochs=epochs, patience=patience)
        self.time_model.fit(epochs=epochs, patience=patience)
        self.kills_model.fit(epochs=epochs, patience=patience)
        self.bosses_model.fit(epochs=epochs, patience=patience)

    def predict_match_outcome(self, inputs):
        """Probability that side 0 wins."""
        scores = self.win_model.predict(inputs)
        return torch.sigmoid(torch.from_numpy(scores)).numpy()

    def predict_time(self, inputs):
        """Match length in seconds."""
        return self.time_model.predict(inputs) * SECONDS_PER_MINUTE

    def predict_kills(self, inputs):
        """Predicted kills for sides 0 and 1."""
        return self.kills_model.predict(inputs)

    def predict_objective(self, inputs):
        """Predicted mid-bosses claimed by sides 0 and 1."""
        return self.bosses_model.predict(inputs)
