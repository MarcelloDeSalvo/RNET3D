"""Transfer stopped four-class TBI checkpoints into an unchanged MouseFull graph."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from test_experiments import experiment
from src.experiments.config import ExperimentConfig, InitializationConfig, LabelConfig
from src.experiments.initialization import backbone, graph_signature, initialize_model
from src.experiments.training import build_model, build_network, training_view


def mousefull_labels():
    """Keep the production source/channel order in the miniature transfer test."""
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 12: 6, 13: 7, 15: 8, 16: 9, 21: 10}
    names = tuple(str(source) for source in mapping)
    return LabelConfig(mapping=mapping, names=names)


def write_tbi_run(experiment, folder, *, roi_loss='dice_ce'):
    """Save committed fitting weights without a model export or optimizer."""
    model_settings = replace(experiment.model, architecture='3drnet', residual=True, attention=True)
    source_config = replace(experiment, labels=LabelConfig(), model=model_settings,
                            loss=replace(experiment.loss, regions=roi_loss))
    source = build_network(source_config)
    features = backbone(source)
    for index, variable in enumerate(features.weights):
        value = (index + 1) / 1000
        variable.assign(tf.fill(variable.shape, tf.cast(value, variable.dtype)))
    source = training_view(source, source_config)
    fold = folder / 'fold_0'
    latest = fold / 'latest'
    latest.mkdir(parents=True)
    checkpoint = latest / 'selected_1.weights.h5'
    source.save_weights(str(checkpoint))
    # An incomplete public save must not override committed selected weights.
    (fold / 'best.weights.h5').write_bytes(b'interrupted write')
    state = {'selected_checkpoint': 'latest/selected_1.weights.h5', 'completed_epochs': 1, 'fit_complete': False}
    (fold / 'training_state.json').write_text(json.dumps(state), encoding='utf-8')
    (folder / 'experiment.json').write_text(json.dumps(source_config.to_dict()), encoding='utf-8')
    return source_config, source, checkpoint


@pytest.mark.parametrize('roi_loss', ['dice_ce', 'dice_ce_gradient_fix'])
@pytest.mark.parametrize('freeze', [False, True])
def test_stopped_tbi_transfer_copies_all_features_and_keeps_every_new_head(
    experiment, tmp_path, monkeypatch, roi_loss, freeze,
):
    source_config, source, checkpoint = write_tbi_run(experiment, tmp_path / 'tbi', roi_loss=roi_loss)
    settings = InitializationConfig.from_run(tmp_path / 'tbi', freeze_backbone=freeze)
    assert settings.checkpoint == checkpoint.resolve()
    assert settings.source_experiment == (tmp_path / 'tbi' / 'experiment.json').resolve()
    target_config = replace(source_config, labels=mousefull_labels(),
                            model=replace(source_config.model, classifier=True))
    target = build_network(target_config)
    target_features = backbone(target)
    feature_names = {layer.name for layer in target_features.layers}
    fresh_heads = {
        layer.name: layer.get_weights() for layer in target.layers
        if layer.weights and layer.name not in feature_names
    }

    def forbidden(*args, **kwargs):
        pytest.fail('Loading the source checkpoint must not compile or fit it')

    monkeypatch.setattr(tf.keras.Model, 'compile', forbidden)
    monkeypatch.setattr(tf.keras.Model, 'fit', forbidden)
    report = initialize_model(target, settings)
    assert target.output_names == ['regions', 'brain_mask', 'classifier']
    assert target.output_shape[0][-1] == 11
    for actual, expected in zip(target_features.get_weights(), backbone(source).get_weights()):
        np.testing.assert_array_equal(actual, expected)
    for name, weights in fresh_heads.items():
        for actual, expected in zip(target.get_layer(name).get_weights(), weights):
            np.testing.assert_array_equal(actual, expected)
    assert set(report['reinitialized_heads']) == set(fresh_heads)
    assert all(layer.trainable != freeze for layer in target_features.layers)
    assert all(target.get_layer(name).trainable for name in fresh_heads)
    assert report['source_labels']['mapping'] == {0: 0, 1: 1, 3: 2, 13: 2, 21: 3}
    assert report['source_experiment_sha256']


def test_transfer_mismatch_fails_before_changing_target_weights(experiment, tmp_path):
    source_config, _, _ = write_tbi_run(experiment, tmp_path / 'tbi')
    settings = InitializationConfig.from_run(tmp_path / 'tbi')
    target_config = replace(source_config, model=replace(source_config.model, attention=False))
    target = build_network(target_config)
    before = target.get_weights()
    with pytest.raises(ValueError, match='backbone graphs differ'):
        initialize_model(target, settings)
    for actual, expected in zip(target.get_weights(), before):
        np.testing.assert_array_equal(actual, expected)


def test_fresh_mousefull_models_share_pretrained_features_with_fresh_optimizers(experiment, tmp_path):
    source_config, source, _ = write_tbi_run(experiment, tmp_path / 'tbi')
    settings = InitializationConfig.from_run(tmp_path / 'tbi')
    config = replace(source_config, labels=mousefull_labels(), initialization=settings,
                     model=replace(source_config.model, classifier=True),
                     loss=replace(source_config.loss, regions='dice_ce_gradient_fix'))
    for _ in range(2):
        target = build_model(config)
        assert target.output_names == ['regions_logits', 'brain_mask', 'classifier']
        assert target.output_shape[0][-1] == 11
        assert int(target.optimizer.iterations.numpy()) == 0
        for actual, expected in zip(backbone(target).get_weights(), backbone(source).get_weights()):
            np.testing.assert_array_equal(actual, expected)


def test_initialization_paths_roundtrip_and_missing_source_errors(experiment, tmp_path):
    source_config, _, checkpoint = write_tbi_run(experiment, tmp_path / 'tbi')
    settings = InitializationConfig.from_run(tmp_path / 'tbi')
    config = replace(source_config, initialization=settings)
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    old_settings = source_config.to_dict()
    del old_settings['initialization']['source_experiment']
    assert ExperimentConfig.from_dict(old_settings).initialization.source_experiment is None
    with pytest.raises(FileNotFoundError):
        InitializationConfig.from_run(tmp_path / 'missing')
    with pytest.raises(ValueError, match='Source fold'):
        InitializationConfig.from_run(tmp_path / 'tbi', fold=-1)
    invalid = InitializationConfig(mode='weights', checkpoint=checkpoint, source_experiment=settings.source_experiment)
    with pytest.raises(ValueError, match='only used for backbone'):
        invalid.validate()


def test_mousefull_notebooks_expose_transfer_and_keep_preview_safe():
    root = Path(__file__).resolve().parents[1] / 'notebook'
    for filename in ('05_mouse_full_training.ipynb', '06_mouse_full_kfold.ipynb'):
        cells = json.loads((root / filename).read_text(encoding='utf-8'))['cells']
        for index, cell in enumerate(cells):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'{filename}:{index}', 'exec')
        config = ''.join(cells[3]['source'])
        assert 'USE_TBI_PRETRAINING: bool = True' in config
        assert 'TBI_RUN: Path | None = None' in config
        assert 'FREEZE_BACKBONE: bool = False' in config
        assert 'InitializationConfig.from_run(' in config
        sources = [''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code']
        assert any('if USE_TBI_PRETRAINING and config.initialization.mode == "scratch":' in source for source in sources)


def test_graph_signature_normalizes_names_without_corrupting_operation_literals():
    """A first Lambda named 'lambda' must compare to later autogenerated names."""
    def repeat(value):
        return value + value

    def model(layer_name):
        image = tf.keras.Input((2,), name=f'input_{layer_name}')
        output = tf.keras.layers.Lambda(repeat, name=layer_name)(image)
        return tf.keras.Model(image, output)

    first = model('lambda')
    second = model('lambda_100')
    assert graph_signature(first, for_transfer=True) == graph_signature(second, for_transfer=True)
    signature = json.loads(graph_signature(first, for_transfer=True))
    assert signature['layers'][1]['config']['function_type'] == 'lambda'


def test_resume_verifies_weights_only_source_configuration(experiment, tmp_path):
    from src.experiments.data import prepare_data
    from src.experiments.training import save_provenance, validate_resume

    source_config, _, _ = write_tbi_run(experiment, tmp_path / 'tbi')
    settings = InitializationConfig.from_run(tmp_path / 'tbi')
    config = replace(source_config, labels=mousefull_labels(), initialization=settings,
                     loss=replace(source_config.loss, regions='dice_ce_gradient_fix'))
    data = prepare_data(config)
    run = tmp_path / 'target'
    run.mkdir()
    save_provenance(config, data, run)
    validate_resume(data, run)
    with settings.source_experiment.open('a', encoding='utf-8') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='initialization source experiment has changed'):
        validate_resume(data, run)


def test_resume_accepts_older_configs_without_the_new_optional_field(experiment, tmp_path):
    from src.experiments.data import prepare_data
    from src.experiments.training import save_provenance, validate_resume

    data = prepare_data(experiment)
    run = tmp_path / 'old_target'
    run.mkdir()
    save_provenance(experiment, data, run)
    path = run / 'experiment.json'
    settings = json.loads(path.read_text(encoding='utf-8'))
    del settings['initialization']['source_experiment']
    path.write_text(json.dumps(settings), encoding='utf-8')
    validate_resume(data, run)
