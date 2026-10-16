"""Interrupt real fitting, restore optimizer progress, and reuse completed folds."""

from dataclasses import replace
import json

import numpy as np
import pytest
import tensorflow as tf

from src.evaluation.k_fold import TrainingConfig, k_fold_loop


@pytest.mark.parametrize("selection", [None, float("nan"), float("inf"), float("-inf")])
def test_invalid_selection_reports_the_epoch_without_advancing_recovery(tmp_path, selection):
    """A failed epoch retains the last committed history and explains why it failed."""
    from src.evaluation.training_state import EpochCheckpoint, write_json

    state = EpochCheckpoint(tmp_path, [], resume=False, monitor="val_loss")
    state.history = {"loss": [0.5], "val_loss": [0.6]}
    state.completed_epochs = 1
    committed = {"completed_epochs": 1, "history": state.history}
    write_json(state.path, committed)
    values = {"loss": 0.4}
    if selection is not None:
        values["val_loss"] = selection

    with pytest.raises(ValueError, match="Epoch 2:.*'val_loss'") as error:
        state.on_epoch_end(1, values)

    assert "Diagnostic report:" in str(error.value)
    assert state.completed_epochs == 1
    assert state.history == committed["history"]
    assert json.loads(state.path.read_text()) == committed
    assert not (tmp_path / "latest").exists()
    report = json.loads((tmp_path / "invalid_epoch_2.json").read_text())
    assert report["epoch"] == 2 and report["last_completed_epoch"] == 1
    assert report["monitor"] == "val_loss"
    if selection is None:
        assert report["reason"] == "missing"
    else:
        assert report["reason"] == "nonfinite"
        assert report["logs"]["val_loss"] == repr(selection)


def test_invalid_selection_reports_nonfinite_batchnorm_statistics(tmp_path):
    """Distinguish broken inference statistics from an absent validation metric."""
    from src.evaluation.training_state import EpochCheckpoint

    image = tf.keras.Input((1,))
    normalization = tf.keras.layers.BatchNormalization()
    model = tf.keras.Model(image, normalization(image))
    model.compile(optimizer="sgd", loss="mse")
    normalization.moving_variance.assign([float("nan")])
    state = EpochCheckpoint(tmp_path, [], resume=False, monitor="val_loss")
    state.set_model(model)

    with pytest.raises(ValueError, match="'val_loss' is nan"):
        state.on_epoch_end(0, {"loss": 0.5, "val_loss": float("nan")})

    report = json.loads((tmp_path / "invalid_epoch_1.json").read_text())
    assert report["nonfinite_weights"] == [normalization.moving_variance.name]
    assert not state.path.exists()


def test_interrupted_cv_restores_optimizer_and_skips_completed_folds(tmp_path):
    control = {"interrupt": True}
    initial_iterations = []
    builds = []

    def dataset(ids):
        inputs = np.asarray([[int(subject)] for subject in ids], dtype=np.float32)
        targets = inputs * 0.5
        data = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(1)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return data.with_options(options)

    def model():
        builds.append(True)
        inputs = tf.keras.Input((1,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        result = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        result.compile(optimizer=optimizer, loss="mse")
        return result

    class Interrupt(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            initial_iterations.append(int(self.model.optimizer.iterations.numpy()))

        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 1 and control["interrupt"]:
                raise RuntimeError("Simulated interruption")

    def callbacks(context):
        return [tf.keras.callbacks.EarlyStopping(patience=10), Interrupt()]

    config = TrainingConfig(
        model_name="recovery", run_id="run", output_dir=str(tmp_path), k=2,
        epochs=3, batch_size=1, fit_kwargs={"verbose": 0},
    )
    ids = ["1", "2", "3", "4"]
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        k_fold_loop(dataset, dataset, model, config, ids, callbacks_factory=callbacks)

    folder = tmp_path / "recovery" / "run"
    record = json.loads((folder / "fold_0" / "training_state.json").read_text())
    assert record["completed_epochs"] == 1
    assert not record["fit_complete"]
    # A crash after best-weight writing but before state commit must roll back
    # that file to the selected weights paired with the committed epoch.
    (folder / "fold_0" / "best.weights.h5").write_bytes(b"uncommitted checkpoint")
    control["interrupt"] = False
    result = k_fold_loop(
        dataset, dataset, model, config, ids, callbacks_factory=callbacks, resume_run=folder,
    )
    assert initial_iterations == [0, 2, 0]
    assert [len(history["loss"]) for history in result.histories] == [3, 3]
    assert len(builds) == 3

    cached = k_fold_loop(
        dataset, dataset, model, config, ids, callbacks_factory=callbacks, resume_run=folder,
    )
    assert len(builds) == 3
    assert cached.histories == result.histories

    reference_config = replace(config, model_name="reference")
    reference = k_fold_loop(dataset, dataset, model, reference_config, ids, callbacks_factory=callbacks)
    for fold in range(2):
        restored = tf.keras.models.load_model(folder / f"fold_{fold}" / "model.keras")
        uninterrupted = tf.keras.models.load_model(reference.run_dir / f"fold_{fold}" / "model.keras")
        assert int(restored.optimizer.iterations.numpy()) == 6
        for recovered, expected in zip(restored.get_weights(), uninterrupted.get_weights()):
            np.testing.assert_allclose(recovered, expected, rtol=1e-6, atol=1e-6)

    changed = replace(config, epochs=4)
    with pytest.raises(ValueError, match="configuration differs"):
        k_fold_loop(dataset, dataset, model, changed, ids, callbacks_factory=callbacks, resume_run=folder)


def test_run_lock_rejects_a_second_writer(tmp_path):
    from src.evaluation.training_state import run_lock

    with run_lock(tmp_path):
        with pytest.raises((RuntimeError, BlockingIOError)):
            with run_lock(tmp_path):
                pytest.fail("A second writer acquired the same run")


def test_resume_can_add_regions_dice_without_fabricating_previous_scores(tmp_path):
    """An existing run retains optimizer progress when adding a training monitor."""
    from src.evaluation.metrics import PartialLabelDice

    control = {"interrupt": True, "add_metric": False}
    iterations = []

    def dataset(ids):
        values = np.asarray([int(subject) for subject in ids])
        inputs = values.astype(np.float32).reshape(-1, 1, 1, 1, 1)
        targets = tf.one_hot(values % 3, 3).numpy().reshape(-1, 1, 1, 1, 3)
        return tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(1)

    def model():
        inputs = tf.keras.Input((1, 1, 1, 1))
        outputs = tf.keras.layers.Conv3D(3, 1, name="regions_logits")(inputs)
        result = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = []
        if control["add_metric"]:
            metrics.append(PartialLabelDice(num_classes=3))
        result.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return result

    class Interrupt(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            iterations.append(int(self.model.optimizer.iterations.numpy()))

        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 1 and control["interrupt"]:
                raise RuntimeError("Add Dice before resuming")

    config = TrainingConfig(
        model_name="dice_monitor", run_id="run", output_dir=str(tmp_path), k=2,
        epochs=2, batch_size=1, default_callbacks=False, fit_kwargs={"verbose": 0},
    )
    ids = ["1", "2", "3", "4"]
    def callbacks(context):
        return [Interrupt()]

    with pytest.raises(RuntimeError, match="Add Dice"):
        k_fold_loop(dataset, dataset, model, config, ids, callbacks_factory=callbacks)

    control.update(interrupt=False, add_metric=True)
    folder = tmp_path / "dice_monitor" / "run"
    result = k_fold_loop(
        dataset, dataset, model, config, ids, callbacks_factory=callbacks, resume_run=folder,
    )
    assert iterations == [0, 2, 0]
    for history in result.histories:
        assert all(len(values) == 2 for values in history.values())
    resumed_history = result.histories[0]
    for key in ("mean_dice", "val_mean_dice"):
        assert resumed_history[key][0] is None
        assert 0 <= resumed_history[key][1] <= 1
