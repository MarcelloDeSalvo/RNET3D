"""Epoch-boundary recovery of model, optimizer, history and standard callbacks.

Import this module only after runtime configuration: its callback uses TensorFlow.
Recovery restarts an interrupted epoch and does not promise identical stochastic
augmentation or accelerator results to an uninterrupted run.
"""

from __future__ import annotations

import json
import hashlib
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf


def write_json(path: Path, payload: Any) -> None:
    """Replace one state file atomically after its dependent artifacts are written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def run_lock(folder: Path) -> Iterator[None]:
    """Prevent simultaneous writers; the OS releases the lock on process exit."""
    import os

    path = folder / ".run.lock"
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("This run is already being written by another process") from error
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class EpochCheckpoint(tf.keras.callbacks.Callback):
    """Keep a latest fitting checkpoint separate from selected best weights."""

    def __init__(
        self, folder: Path, callbacks: list[Any], *, resume: bool, monitor: str,
    ) -> None:
        """Read the committed recovery record, never an uncommitted latest file."""
        super().__init__()
        self.folder: Path = folder
        self.path: Path = folder / "training_state.json"
        self.callbacks: list[Any] = callbacks
        self.monitor: str = monitor
        self.record: dict[str, Any] = {}
        self.history: dict[str, list[float | None]] = {}
        self.completed_epochs: int = 0
        self.training_seconds: float = 0.0
        self.fit_complete: bool = False
        self.started: float = 0.0
        if resume and self.path.exists():
            self.record = json.loads(self.path.read_text(encoding="utf-8"))
            self.history = self.record["history"]
            self.completed_epochs = self.record["completed_epochs"]
            self.training_seconds = self.record["training_seconds"]
            self.fit_complete = self.record["fit_complete"]

    def set_model(self, model: Any) -> None:
        """Bind an already compiled graph and restore all optimizer slots."""
        if self.model is model and hasattr(self, "checkpoint"):
            return
        super().set_model(model)
        from ..experiments.initialization import graph_signature

        signature = graph_signature(model)
        self.graph_sha256: str = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        if self.record and self.record["graph_sha256"] != self.graph_sha256:
            raise ValueError("Resume model graph differs from the saved fitting graph")
        optimizer = model.optimizer
        if hasattr(optimizer, "_create_all_weights"):
            optimizer._create_all_weights(model.trainable_variables)
        elif hasattr(optimizer, "build"):
            optimizer.build(model.trainable_variables)
        self.checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, directory=str(self.folder / "latest"), max_to_keep=2,
        )
        if self.record:
            path = str(self.folder / self.record["checkpoint"])
            status = self.checkpoint.restore(path)
            status.assert_consumed()
            selected = self.folder / self.record["selected_checkpoint"]
            shutil.copyfile(selected, self.folder / "best.weights.h5")

    def _callback_states(self) -> list[dict[str, Any]]:
        """Capture checkpoint selection, early stopping and LR-reduction state."""
        states = []
        fields = {
            "ModelCheckpoint": ("best",),
            "EarlyStopping": ("best", "wait", "stopped_epoch", "best_epoch"),
            "ReduceLROnPlateau": ("best", "wait", "cooldown_counter"),
        }
        for callback in self.callbacks:
            name = type(callback).__name__
            values = {}
            for field in fields.get(name, ()):
                if hasattr(callback, field):
                    value = getattr(callback, field)
                    if isinstance(value, np.generic):
                        value = value.item()
                    values[field] = value
            states.append({"class": name, "values": values})
        return states

    def on_train_begin(self, logs: Any = None) -> None:
        """Restore callback state after their normal initialization hooks."""
        print(f"Training checkpoints: {self.folder}")
        if not self.record:
            return
        saved = self.record["callbacks"]
        if [entry["class"] for entry in saved] != [type(callback).__name__ for callback in self.callbacks]:
            raise ValueError("Resume requires the same callback classes and ordering")
        for callback, entry in zip(self.callbacks, saved):
            for field, value in entry["values"].items():
                setattr(callback, field, value)

        state = self.record["numpy_random_state"]
        array = np.asarray(state[1], dtype=np.uint32)
        np.random.set_state((state[0], array, state[2], state[3], state[4]))

        def tuples(value: Any) -> Any:
            """Restore nested Python random-state tuples from JSON arrays."""
            if isinstance(value, list):
                return tuple(tuples(item) for item in value)
            return value

        random.setstate(tuples(self.record["python_random_state"]))

    def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
        """Measure completed epoch time, including its validation pass."""
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        """Commit latest state after the best-checkpoint callback has finished."""
        values = {key: float(value) for key, value in (logs or {}).items()}
        if self.monitor not in values or not np.isfinite(values[self.monitor]):
            raise ValueError("Every epoch needs a finite checkpoint-selection metric")
        # Newly compiled monitors have no observations for earlier epochs.
        for key in values:
            if key not in self.history:
                self.history[key] = [None] * epoch

        for key, observations in self.history.items():
            observations.append(values.get(key))
        self.completed_epochs = epoch + 1
        self.training_seconds += time.perf_counter() - self.started
        checkpoint_path = self.manager.save(checkpoint_number=self.completed_epochs)
        callback_states = self._callback_states()
        managed_path = (self.folder / "best.weights.h5").resolve()
        managed_callback: tf.keras.callbacks.ModelCheckpoint | None = None
        for callback in self.callbacks:
            if not isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                continue
            callback_path = Path(callback.filepath).resolve()
            if callback_path == managed_path:
                managed_callback = callback
                break
        if managed_callback is None:
            raise ValueError("Epoch recovery requires the managed best-weights checkpoint")
        best = float(managed_callback.best)
        previous_best = self.record.get("selected_metric")
        selected_checkpoint = self.record.get("selected_checkpoint")
        if selected_checkpoint is None or best != previous_best:
            selected_path = self.folder / "latest" / f"selected_{self.completed_epochs}.weights.h5"
            shutil.copyfile(self.folder / "best.weights.h5", selected_path)
            selected_checkpoint = str(selected_path.relative_to(self.folder))

        numpy_state = np.random.get_state()
        self.record = {
            "checkpoint": str(Path(checkpoint_path).relative_to(self.folder)),
            "graph_sha256": self.graph_sha256,
            "completed_epochs": self.completed_epochs, "history": self.history,
            "training_seconds": self.training_seconds, "fit_complete": bool(self.model.stop_training),
            "selected_checkpoint": selected_checkpoint, "selected_metric": best,
            "callbacks": callback_states, "python_random_state": random.getstate(),
            "numpy_random_state": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        }
        write_json(self.path, self.record)
        for path in (self.folder / "latest").glob("selected_*.weights.h5"):
            if str(path.relative_to(self.folder)) != selected_checkpoint:
                path.unlink()

    def on_train_end(self, logs: Any = None) -> None:
        """Mark normally completed fitting so interrupted evaluation can be retried."""
        self.fit_complete = True
        if self.record:
            self.record["fit_complete"] = True
            write_json(self.path, self.record)
