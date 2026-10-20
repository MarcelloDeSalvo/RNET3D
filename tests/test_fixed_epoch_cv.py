"""Fixed-budget rat CV never fits on holdouts or selects epochs from their scores."""

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.evaluation.k_fold import (
    EvaluationConfig, FoldContext, FoldSplit, TrainingConfig, _build_callbacks, k_fold_loop,
)
from src.evaluation.inference import best_checkpoint_settings
from src.experiments.training import history_figure
from src.experiments.schedules import CosineEpochSchedule


def test_fixed_epoch_cv_uses_final_weights_without_validation_and_resumes(tmp_path):
    """Interrupt and recover real fits, then inspect final patch/volume evaluators."""
    config = TrainingConfig(
        model_name="fixed_rats", run_id="run", output_dir=str(tmp_path), k=2,
        epochs=3, monitor="loss", fixed_epochs=True, default_callbacks=False,
        save_model=False, fit_kwargs={"verbose": 0},
    )
    folds = [FoldSplit(0, ["a"], [], ["b"]), FoldSplit(1, ["b"], [], ["a"])]
    control = {"interrupt": True}
    builds = []
    evaluations = []
    patch_calls = []

    def dataset(subject_ids):
        shape = (len(subject_ids), 1)
        inputs = np.ones(shape, dtype=np.float32)
        targets = np.zeros(shape, dtype=np.float32)
        result = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(1)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return result.with_options(options)

    def forbidden_validation(subject_ids):
        pytest.fail("The held-out rat must never be supplied to fitting-time validation")

    def model():
        builds.append(True)
        result = tf.keras.Sequential([
            tf.keras.layers.Input((1,)),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="zeros"),
        ])
        result.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss="mse")
        return result

    class ControlledEpoch(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 1 and control["interrupt"]:
                raise RuntimeError("Interrupted fixed fit")

        def on_epoch_end(self, epoch, logs=None):
            weight = np.array([[epoch + 1.0]], dtype=np.float32)
            self.model.layers[-1].set_weights([weight])
            # Equal final losses must still advance the committed weight snapshot.
            logs["loss"] = (2.0, 1.0, 1.0)[epoch]

    class Evaluator:
        def __init__(self, model, subject_ids, options):
            assert options["dataset_role"] == "holdout"
            self.ids = list(subject_ids)
            self.value = float(model.get_weights()[0].item())
            evaluations.append((self.ids, self.value))

        def evaluate(self):
            return pd.DataFrame({"Subject": self.ids, "Metric": "selected_weight", "Value": self.value})

    def patch_evaluator(model, subject_ids, options):
        assert "holdout" in str(options["evaluation_dir"])
        value = float(model.get_weights()[0].item())
        patch_calls.append((list(subject_ids), value))
        return pd.DataFrame({"Metric": ["selected_weight"], "Value": [value]})

    schedule = CosineEpochSchedule(initial_learning_rate=0.001, minimum_learning_rate=0.0001, epochs=3)
    options = {
        "fold_splits": folds, "callbacks_factory": lambda context: [
            tf.keras.callbacks.LearningRateScheduler(schedule), ControlledEpoch(),
        ],
        "evaluation": EvaluationConfig(evaluate_validation=False),
        "fve_callbacks": {"regions": Evaluator}, "patch_evaluator": patch_evaluator,
    }
    with pytest.raises(RuntimeError, match="Interrupted fixed fit"):
        k_fold_loop(dataset, forbidden_validation, model, config, ["a", "b"], **options)
    run = tmp_path / "fixed_rats/run"
    state_file = run / "fold_0/training_state.json"
    state = json.loads(state_file.read_text())
    assert state["completed_epochs"] == 1 and not state["fit_complete"]
    assert "val_loss" not in state["history"]
    assert not evaluations and not patch_calls
    (run / "fold_0/best.weights.h5").write_bytes(b"Uncommitted overwrite")

    control["interrupt"] = False
    result = k_fold_loop(dataset, forbidden_validation, model, config, ["a", "b"], resume_run=run, **options)
    assert evaluations == patch_calls == [(["b"], 3.0), (["a"], 3.0)]
    assert len(builds) == 3
    assert all(len(history["loss"]) == 3 and "val_loss" not in history for history in result.histories)
    for history in result.histories:
        assert history["lr"] == pytest.approx([0.001, 0.00055, 0.0001])
    assert result.summaries["loss"] == 1.0
    assert result.patch_metrics.BestEpoch.eq(3).all()
    assert result.patch_metrics.DatasetRole.eq("holdout").all()
    assert len(history_figure(result).data) == 2
    for fold in (0, 1):
        state = json.loads((run / f"fold_{fold}/training_state.json").read_text())
        assert Path(state["selected_checkpoint"]).as_posix() == "latest/selected_3.weights.h5"
        assert state["selected_metric"] == 1.0

    cached = k_fold_loop(dataset, forbidden_validation, model, config, ["a", "b"], resume_run=run, **options)
    assert cached.histories == result.histories
    assert len(builds) == 3 and len(evaluations) == 2
    changed = replace(config, fixed_epochs=False)
    with pytest.raises(ValueError, match="configuration differs"):
        k_fold_loop(dataset, forbidden_validation, model, changed, ["a", "b"], resume_run=run, **options)


@pytest.mark.parametrize("callback", [
    tf.keras.callbacks.EarlyStopping(), tf.keras.callbacks.ReduceLROnPlateau(),
])
def test_fixed_epoch_policy_rejects_adaptive_callbacks(tmp_path, callback):
    config = TrainingConfig(model_name="fixed", monitor="loss", fixed_epochs=True, default_callbacks=False)
    context = FoldContext(
        split=FoldSplit(0, ["train"], [], ["test"]), output_dir=tmp_path,
        config=config, checkpoint_path=tmp_path / "best.weights.h5",
    )
    with pytest.raises(ValueError, match="cannot use early stopping or ReduceLROnPlateau"):
        _build_callbacks(context, lambda context: [callback])


def test_fixed_checkpoint_evaluation_requires_the_complete_budget(experiment, tmp_path):
    training = replace(experiment.training, epochs=3, monitor="loss", fixed_epochs=True, default_callbacks=False)
    config = replace(experiment, training=training)
    run = tmp_path / "run"
    fold = run / "fold_0"
    latest = fold / "latest"
    latest.mkdir(parents=True)
    (run / "experiment.json").write_text(json.dumps(config.to_dict()))
    split = {"train": ["D1/SUBJ_0000", "D1/SUBJ_0001", "D1/SUBJ_0002"], "validation": [], "test": ["D1/SUBJ_0003"]}
    (run / "splits.json").write_text(json.dumps([split]))
    (fold / "split.json").write_text(json.dumps(split))
    selected = latest / "selected_3.weights.h5"
    selected.write_bytes(b"Only checkpoint selection is exercised")
    state = {"completed_epochs": 2, "selected_checkpoint": "latest/selected_3.weights.h5"}
    state_file = fold / "training_state.json"
    state_file.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Complete the fixed epoch budget"):
        best_checkpoint_settings(run, role="holdout")
    state["completed_epochs"] = 3
    state_file.write_text(json.dumps(state))
    settings = best_checkpoint_settings(run, role="holdout")
    assert settings.model_file == selected
    assert settings.subject_ids == ("D1/SUBJ_0003",)
