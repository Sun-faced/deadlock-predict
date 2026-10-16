"""Example main: make sure both datasets exist, then load them.

The first run builds what's missing (asking Y/N before downloading). Later runs
find the datasets already on disk and go straight to loading them.
"""

from data import ensure_dataset
from models.basic import Baseline
from models.linear import LinearModel
from models.mlp import MLPModel, TaskMLP
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
    baseline_results = "results/baseline"
    baseline_test = TestModel(model_basic)
    baseline_test.training_accuracy(train_folder / "inputs.npy", train_folder / "outputs.npy",
                                    save_path=f"{baseline_results}/training_accuracy.json")
    baseline_test.test(test_folder / "inputs.npy", test_folder / "outputs.npy",
                       out_dir=baseline_results)

    """
    Start the testing of linear model
    """
    # Turn feature groups on or off to compare them on the same test set.
    linear_features = {
        "ID": True,
        "LANES": True,
        "PARTIES": True,
        "EXPERIENCE": True,
        "FLAGS": True,
    }
    linear_epochs = 5  # Try 30 as a limit when all groups are enabled.
    linear_patience = 6  # Train on all unranked matches for every epoch.
    selected_groups = []
    for name, enabled in linear_features.items():
        if enabled:
            selected_groups.append(name.lower())
    run_name = "_".join(selected_groups) if selected_groups else "none"
    print(f"Linear model features: {run_name}")

    model_linear = LinearModel(train_folder, features=linear_features)
    model_linear.fit(epochs=linear_epochs, patience=linear_patience)
    linear_results = f"results/linear/{run_name}"
    linear_test = TestModel(model_linear)
    linear_test.training_accuracy(train_folder / "inputs.npy", train_folder / "outputs.npy",
                                  save_path=f"{linear_results}/training_accuracy.json")
    linear_test.test(test_folder / "inputs.npy", test_folder / "outputs.npy",
                     out_dir=linear_results)

    """
    Train four independent MLPs with all 63 inputs, then test their outputs
    """
    mlp_epochs = 500  # Maximum for each of the four tasks.
    mlp_patience = 501  # Train on all unranked matches for every epoch.
    win_mlp = TaskMLP("win", train_folder)
    win_mlp.fit(epochs=mlp_epochs, patience=mlp_patience)

    time_mlp = TaskMLP("time", train_folder)
    time_mlp.fit(epochs=mlp_epochs, patience=mlp_patience)

    kills_mlp = TaskMLP("kills", train_folder)
    kills_mlp.fit(epochs=mlp_epochs, patience=mlp_patience)

    bosses_mlp = TaskMLP("bosses", train_folder)
    bosses_mlp.fit(epochs=mlp_epochs, patience=mlp_patience)

    model_mlp = MLPModel(win_model=win_mlp, time_model=time_mlp,
                         kills_model=kills_mlp, bosses_model=bosses_mlp)
    mlp_results = "results/mlp/independent"
    mlp_test = TestModel(model_mlp)
    mlp_test.training_accuracy(train_folder / "inputs.npy", train_folder / "outputs.npy",
                               save_path=f"{mlp_results}/training_accuracy.json")
    mlp_test.test(test_folder / "inputs.npy", test_folder / "outputs.npy",
                  out_dir=mlp_results)
