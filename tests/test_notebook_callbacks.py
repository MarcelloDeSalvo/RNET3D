"""Exercise explicit notebook callbacks, independent fold state and recovery."""

import json
import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.evaluation.k_fold import FoldContext, FoldSplit, _build_callbacks
from src.experiments.config import InferenceConfig
from src.experiments.data import SubjectSplit, make_cv_splits, prepare_data
from src.experiments.training import (
    _copy_callbacks, callback_configuration, run_cross_validation, run_training,
)


def test_callbacks_are_copied_with_separate_fold_paths_and_exact_options(experiment, tmp_path):
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir='tensorboard', histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(patience=20, mode='min', restore_best_weights=False),
        tf.keras.callbacks.CSVLogger('logs/training.csv', append=True),
    ]
    clone_lists = []
    for index in (0, 1):
        folder = tmp_path / f'fold_{index}'
        context = FoldContext(
            split=FoldSplit(index=index, train_ids=['train'], val_ids=['val']),
            output_dir=folder, config=experiment.training, checkpoint_path=folder / 'best.weights.h5',
        )
        cloned = _build_callbacks(context, lambda context: _copy_callbacks(callbacks, context))
        clone_lists.append(cloned)
        assert [type(callback).__name__ for callback in cloned] == [
            'ReduceLROnPlateau', 'TensorBoard', 'EarlyStopping', 'CSVLogger', 'ModelCheckpoint',
        ]
        assert cloned[0].min_lr == 1e-7
        assert cloned[1].histogram_freq == 1
        assert cloned[1].log_dir == str(folder / 'tensorboard')
        assert cloned[3].filename == str(folder / 'logs/training.csv')
        assert Path(cloned[3].filename).parent.is_dir()
    assert clone_lists[0][0] is not clone_lists[1][0]
    clone_lists[0][0].wait = 9
    assert clone_lists[1][0].wait == callbacks[0].wait == 0
    assert callbacks[1].log_dir == 'tensorboard'
    assert callbacks[3].filename == 'logs/training.csv'
    assert all(callback.model is None for callback in callbacks)


def test_explicit_callbacks_resume_optimizer_lr_state_and_csv_without_mutating_templates(experiment):
    config = replace(experiment, training=replace(experiment.training, epochs=2, batch_size=1),
                     inference=InferenceConfig(enabled=False, stride=16))
    data = prepare_data(config)
    split = SubjectSplit(train=('D1/SUBJ_0000', 'D1/SUBJ_0001'),
                         validation=('D1/SUBJ_0002',), test=('D1/SUBJ_0003',))
    control = {'interrupt': True}

    class Interrupt(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 1 and control['interrupt']:
                raise RuntimeError('stop after one saved epoch')

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=1, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger('training.csv', append=True), Interrupt(),
    ]
    configuration = callback_configuration(callbacks)
    with pytest.raises(RuntimeError, match='saved epoch'):
        run_training(data, split, callbacks=callbacks)
    run = next((Path(config.training.output_dir) / config.training.model_name).iterdir())
    state = json.loads((run / 'fold_0/training_state.json').read_text())
    assert state['completed_epochs'] == 1
    assert json.loads((run / 'callbacks.json').read_text()) == configuration
    control['interrupt'] = False
    result = run_training(data, split, resume_run=run, callbacks=callbacks)
    assert len(result.histories[0]['loss']) == 2
    lines = (run / 'fold_0/training.csv').read_text().splitlines()
    assert len(lines) == 3 and lines[1].startswith('0,') and lines[2].startswith('1,')
    assert all(callback.model is None for callback in callbacks)
    assert callback_configuration(callbacks) == configuration
    callbacks[0].patience = 2
    with pytest.raises(ValueError, match='callback settings differ'):
        run_training(data, split, resume_run=run, callbacks=callbacks)


def test_cv_runs_full_budget_without_implicit_early_stopping(experiment):
    config = replace(experiment, training=replace(experiment.training, k=2, epochs=2, batch_size=1),
                     inference=InferenceConfig(enabled=False, stride=16), early_stopping_patience=0)
    data = prepare_data(config)
    folds = make_cv_splits(data.subjects.subject_ids, k=2, seed=42)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)]
    result = run_cross_validation(data, folds=folds, callbacks=callbacks)
    assert all(len(history['loss']) == 2 for history in result.histories)
    for index in (0, 1):
        state = json.loads((result.run_dir / f'fold_{index}/training_state.json').read_text())
        assert [entry['class'] for entry in state['callbacks']] == [
            'ReduceLROnPlateau', 'ModelCheckpoint', 'EpochCheckpoint',
        ]
    assert callbacks[0].model is None and callbacks[0].wait == 0 and np.isinf(callbacks[0].best)
    callbacks[0].min_lr = 1e-6
    with pytest.raises(ValueError, match='callback settings differ'):
        run_cross_validation(data, folds=folds, callbacks=callbacks, resume_run=result.run_dir)


def test_three_way_cv_reduces_lr_on_inner_validation_and_scores_selected_weights(tmp_path):
    """Outer scores use the best validation epoch, even if training improves later."""
    from src.evaluation.k_fold import EvaluationConfig, TrainingConfig, k_fold_loop

    folds = [
        FoldSplit(0, ['a'], ['b'], ['c']),
        FoldSplit(1, ['b'], ['c'], ['a']),
        FoldSplit(2, ['c'], ['a'], ['b']),
    ]
    final_learning_rates = []
    evaluated = []

    def dataset(subject_ids):
        """Supply one deterministic batch for each fitting or validation animal."""
        shape = (len(subject_ids), 1)
        inputs = np.ones(shape, dtype=np.float32)
        targets = np.zeros(shape, dtype=np.float32)
        result = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(1)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return result.with_options(options)

    def model():
        """Use a single weight so the selected epoch is directly observable."""
        result = tf.keras.Sequential([
            tf.keras.layers.Input((1,)),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='zeros'),
        ])
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        result.compile(optimizer=optimizer, loss='mse')
        return result

    class ControlledEpoch(tf.keras.callbacks.Callback):
        """Give the three epochs distinct weights and conflicting loss curves."""

        def on_epoch_end(self, epoch, logs=None):
            weight = np.array([[epoch + 1.0]], dtype=np.float32)
            self.model.layers[-1].set_weights([weight])
            logs['loss'] = (0.9, 0.6, 0.1)[epoch]
            logs['val_loss'] = (0.1, 0.3, 0.4)[epoch]

        def on_train_end(self, logs=None):
            rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            final_learning_rates.append(rate)

    def callbacks(context):
        """Reduce only on inner validation; keep all three fitting epochs."""
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', mode='min', factor=0.5, patience=1, min_lr=1e-7,
        )
        return [ControlledEpoch(), reduce_lr]

    class HoldoutEvaluator:
        """Inspect the weights used by the actual post-fit evaluator entry point."""

        def __init__(self, model, subject_ids, options):
            assert options['dataset_role'] == 'holdout'
            expected = folds[options['fold']].test_ids
            assert list(subject_ids) == expected
            self.subject_ids = list(subject_ids)
            self.weight = float(model.get_weights()[0].item())
            evaluated.append((options['fold'], self.subject_ids, self.weight))

        def evaluate(self):
            import pandas as pd

            return pd.DataFrame({
                'Subject': self.subject_ids, 'Metric': 'selected_weight', 'Value': self.weight,
            })

    config = TrainingConfig(
        model_name='three_way_selection', output_dir=str(tmp_path), k=3, epochs=3,
        monitor='val_loss', monitor_mode='min', default_callbacks=False,
        save_model=False, fit_kwargs={'verbose': 0},
    )
    evaluation = EvaluationConfig(evaluate_validation=False)
    result = k_fold_loop(
        dataset, dataset, model, config, ['a', 'b', 'c'],
        fold_splits=folds, callbacks_factory=callbacks,
        fve_callbacks={'regions': HoldoutEvaluator}, evaluation=evaluation,
    )

    assert all(len(history['loss']) == 3 for history in result.histories)
    assert final_learning_rates == pytest.approx([0.0025] * 3)
    assert evaluated == [(0, ['c'], 1.0), (1, ['a'], 1.0), (2, ['b'], 1.0)]
    for index, table in enumerate(result.evaluations['regions']):
        assert table.DatasetRole.eq('holdout').all()
        assert table.BestEpoch.eq(1).all()
        state_file = result.run_dir / f'fold_{index}' / 'training_state.json'
        state = json.loads(state_file.read_text())
        assert state['selected_metric'] == pytest.approx(0.1)


def test_study_forwards_same_explicit_list_and_records_settings(experiment, tmp_path, monkeypatch):
    from src.experiments import ablation
    from src.evaluation.k_fold import TrainingResult
    from src.experiments.data import make_split

    data = prepare_data(experiment)
    split = make_split(data.subjects, seed=42)
    received = []

    def runner(prepared, selection, *, resume_run, callbacks):
        received.append(callbacks)
        path = prepared.config.training.start_run()
        return TrainingResult(run_dir=path)

    monkeypatch.setattr(ablation, 'run_training', runner)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-7)]
    variants = [ablation.AblationVariant('reference'), ablation.AblationVariant('augmented', 'augment', True)]
    folder = tmp_path / 'study'
    ablation.run_ablation(data, variants, study_dir=folder, split=split, callbacks=callbacks)
    assert len(received) == 2 and all(value is callbacks for value in received)
    contract = json.loads((folder / 'study.json').read_text())
    assert contract['callbacks'] == callback_configuration(callbacks)
    callbacks[0].min_lr = 1e-6
    with pytest.raises(ValueError, match='Resume study'):
        ablation.run_ablation(data, variants, study_dir=folder, split=split, callbacks=callbacks, resume=True)


def test_visible_notebook_lists_match_original_single_and_cv_policies():
    root = Path(__file__).resolve().parents[1] / 'notebook'
    for path in root.glob('0*.ipynb'):
        cells = json.loads(path.read_text(encoding='utf-8'))['cells']
        sources = [''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code']
        for index, source in enumerate(sources):
            compile(source, f'{path.name}:{index}', 'exec')
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and any(keyword.arg == 'callbacks' for keyword in node.keywords):
                    function_name = getattr(node.func, 'id', None)
                    assert function_name in ('run_training', 'run_cross_validation', 'run_ablation', 'run_model_comparison')
        configs = [source for source in sources if 'training = TrainingConfig(' in source]
        if not configs:
            continue
        config = configs[0]
        assert 'default_callbacks=False' in config
        source = next(source for source in sources if 'callbacks: list[tf.keras.callbacks.Callback]' in source)
        namespace = {'MODE': 'fixed_split', 'K_FOLD_COMPATIBLE': False}
        exec(source, namespace)
        callbacks = namespace['callbacks']
        if path.name.startswith('04_'):
            assert callbacks == []
            assert 'monitor="loss"' in config
        else:
            assert callbacks[0].factor == 0.5 and callbacks[0].patience == 10 and callbacks[0].min_lr == 1e-7
        if path.name.startswith(('02_', '06_')):
            assert len(callbacks) == 1
        elif not path.name.startswith('04_'):
            assert [type(callback).__name__ for callback in callbacks] == [
                'ReduceLROnPlateau', 'TensorBoard', 'EarlyStopping', 'CSVLogger',
            ]
            assert callbacks[1].histogram_freq == 1
            assert callbacks[2].patience == 20
        if path.name.startswith(('07_', '07f_', '08_')):
            namespace['MODE'] = 'cross_validation'
            exec(source, namespace)
            assert len(namespace['callbacks']) == 1
        if path.name.startswith(('01_', '03_', '05_')):
            namespace['K_FOLD_COMPATIBLE'] = True
            exec(source, namespace)
            assert len(namespace['callbacks']) == 1
        calls = [source for source in sources if any(name in source for name in [
            'result = run_training(', 'result = run_cross_validation(',
            'study = run_ablation(', 'study = run_model_comparison(',
        ])]
        assert len(calls) == 1 and 'callbacks=callbacks' in calls[0]


def test_rat_notebook_checkpoint_and_lr_do_not_depend_on_held_out_loss(tmp_path):
    """Contradictory held-out curves cannot change the selected fitting epoch."""
    from src.evaluation.k_fold import TrainingConfig, _best_epoch
    from src.evaluation.training_state import EpochCheckpoint

    notebook_path = Path(__file__).resolve().parents[1] / 'notebook/04_rat_kfold.ipynb'
    cells = json.loads(notebook_path.read_text(encoding='utf-8'))['cells']
    sources = [''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code']
    config_source = next(source for source in sources if 'training = TrainingConfig(' in source)
    tree = ast.parse(config_source)
    statement = next(node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'training' for target in node.targets))
    namespace = {'TrainingConfig': TrainingConfig, 'TRAINING_SEED': 42, 'REPO_ROOT': tmp_path}
    exec(compile(ast.Module(body=[statement], type_ignores=[]), str(notebook_path), 'exec'), namespace)
    config = namespace['training']
    callback_source = next(source for source in sources if 'callbacks: list[tf.keras.callbacks.Callback]' in source)
    exec(callback_source, namespace)
    assert config.monitor == 'loss' and not config.default_callbacks
    selected_epochs = []
    for index, validation_curve in enumerate(([3.0, 2.0, 0.01], [0.01, 3.0, 2.0])):
        folder = tmp_path / f'case_{index}'
        folder.mkdir()
        context = FoldContext(
            split=FoldSplit(index=index, train_ids=['rat_train'], val_ids=['rat_held_out']),
            output_dir=folder, config=config, checkpoint_path=folder / 'best.weights.h5',
        )
        callbacks = _build_callbacks(context, lambda context: namespace['callbacks'])
        assert [type(callback).__name__ for callback in callbacks] == ['ModelCheckpoint']
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, use_bias=False, input_shape=(1,))])
        model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mse')
        model.stop_training = False
        for callback in callbacks:
            callback.set_model(model)
            callback.on_train_begin()
        state = EpochCheckpoint(folder, callbacks, resume=False, monitor=config.monitor)
        state.set_model(model)
        for epoch, training_loss in enumerate((2.0, 1.0, 1.5)):
            model.layers[0].set_weights([np.array([[epoch + 1.0]], dtype=np.float32)])
            logs = {'loss': training_loss, 'val_loss': validation_curve[epoch]}
            state.on_epoch_begin(epoch)
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
            state.on_epoch_end(epoch, logs)
        selected_epochs.append(_best_epoch(state.history, config.monitor, config.monitor_mode))
        model.load_weights(str(context.checkpoint_path))
        np.testing.assert_array_equal(model.layers[0].get_weights()[0], [[2.0]])
        assert float(tf.keras.backend.get_value(model.optimizer.learning_rate)) == pytest.approx(0.001)
        assert state.record['selected_metric'] == 1.0
    assert selected_epochs == [1, 1]


def test_extra_checkpoint_uses_its_own_file_and_cannot_replace_managed_best(experiment, tmp_path):
    folder = tmp_path / 'fold_0'
    folder.mkdir()
    context = FoldContext(
        split=FoldSplit(index=0, train_ids=['train'], val_ids=['val']), output_dir=folder,
        config=experiment.training, checkpoint_path=folder / 'best.weights.h5',
    )
    extra = tf.keras.callbacks.ModelCheckpoint('extra.weights.h5', save_weights_only=True)
    cloned = _build_callbacks(context, lambda context: _copy_callbacks([extra], context))
    assert cloned[0].filepath == str(folder / 'extra.weights.h5')
    assert cloned[1].filepath == str(context.checkpoint_path)
    duplicate = tf.keras.callbacks.ModelCheckpoint('best.weights.h5')
    with pytest.raises(ValueError, match='different file'):
        _build_callbacks(context, lambda context: _copy_callbacks([duplicate], context))


def test_recovery_uses_managed_selection_when_an_extra_checkpoint_has_another_monitor(tmp_path):
    from src.evaluation.training_state import EpochCheckpoint

    folder = tmp_path / 'fold_0'
    folder.mkdir()
    image = tf.keras.Input((1,))
    output = tf.keras.layers.Dense(1)(image)
    model = tf.keras.Model(image, output)
    model.compile(optimizer='adam', loss='mse')
    model.stop_training = False
    managed_file = folder / 'best.weights.h5'
    model.save_weights(str(managed_file))
    extra = tf.keras.callbacks.ModelCheckpoint(str(folder / 'extra.weights.h5'), monitor='loss')
    managed = tf.keras.callbacks.ModelCheckpoint(str(managed_file), monitor='val_loss')
    extra.best = 0.1
    managed.best = 0.5
    state = EpochCheckpoint(folder, [extra, managed], resume=False, monitor='val_loss')
    state.set_model(model)
    state.on_epoch_begin(0)
    state.on_epoch_end(0, {'val_loss': 0.5, 'loss': 0.1})
    assert state.record['selected_metric'] == 0.5
    assert (folder / state.record['selected_checkpoint']).is_file()


def test_fixed_split_stock_list_writes_tensorboard_histograms_and_csv(experiment):
    config = replace(experiment, inference=InferenceConfig(enabled=False, stride=16))
    data = prepare_data(config)
    split = SubjectSplit(train=('D1/SUBJ_0000', 'D1/SUBJ_0001'),
                         validation=('D1/SUBJ_0002',), test=('D1/SUBJ_0003',))
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
        tf.keras.callbacks.TensorBoard(log_dir='tensorboard', histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=False),
        tf.keras.callbacks.CSVLogger('training.csv', append=True),
    ]
    result = run_training(data, split, callbacks=callbacks)
    folder = result.run_dir / 'fold_0'
    assert (folder / 'training.csv').is_file()
    assert list((folder / 'tensorboard').rglob('events.out.tfevents.*'))
    assert all(callback.model is None for callback in callbacks)
