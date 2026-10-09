"""Check network contracts and the corrected activation, dropout and residual flags."""

import json
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models import networks


SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "network_graphs.json"
SNAPSHOTS = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("snapshot", SNAPSHOTS)
def test_network_options_and_output_contract(snapshot):
    """Exercise each supported option combination against explicit architecture rules."""
    tf.keras.backend.clear_session()
    builder = getattr(networks, snapshot["builder"])
    options = snapshot["options"]
    model = builder((16, 16, 16), 1, 3, [2, 4, 8, 16, 32], **options)

    residual = options.get("residual", snapshot["builder"] == "mt_r_net_3d")
    attention = options.get("attention", snapshot["builder"] == "mt_r_net_3d")
    multiscale = snapshot["builder"] == "mt_r_net_3d"

    expected_parameters = snapshot["parameters"]
    if multiscale and not residual:
        # Three removed 1x1x1 shortcuts each had input_channels*filters + bias.
        expected_parameters -= 2 * (4 + 8 + 16)
    assert model.count_params() == expected_parameters

    expected_additions = 0
    if multiscale:
        expected_additions += 2  # Two deep-supervision fusions.
    if attention:
        expected_additions += 4
    if residual:
        expected_additions += 9  # Four encoders, bottleneck, four decoders.
        if multiscale:
            expected_additions += 3
    additions = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Add)]
    assert len(additions) == expected_additions

    assert model.output_names[:2] == ["regions", "brain_mask"]
    assert model.output_shape[:2] == [(None, 16, 16, 16, 3), (None, 16, 16, 16, 1)]
    assert all(output.dtype == tf.float32 for output in model.outputs)

    if options.get("with_classifier"):
        assert model.output_names[2] == "classifier"
        assert model.output_shape[2] == (None, 1)
    else:
        assert len(model.outputs) == 2

    tf.keras.backend.clear_session()


@pytest.mark.parametrize("activation", ["relu", "sigmoid", True, False, None])
@pytest.mark.parametrize("transpose", [False, True])
def test_convolution_activation_values_and_order(activation, transpose):
    """Check numeric activation behavior for standard and transposed convolutions."""
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(2, 2, 2, 1))
    outputs = networks.conv_block(
        inputs, 1, kernel_size=1, bn=False, activation=activation,
        transpose=transpose, kernel_initializer="ones", use_bias=False,
        kernel_regularizer=None, bias_regularizer=None,
    )
    model = tf.keras.Model(inputs, outputs)
    values = np.linspace(-2, 2, 8, dtype=np.float32).reshape((1, 2, 2, 2, 1))
    actual = model(values, training=False).numpy()

    if activation is True or activation == "relu":
        expected = np.maximum(values, 0)
    elif activation == "sigmoid":
        expected = 1 / (1 + np.exp(-values))
    else:
        expected = values

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    activation_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Activation)]
    if activation is False or activation is None:
        assert not activation_layers
    else:
        assert len(activation_layers) == 1
        assert model.layers[-1] is activation_layers[0]


def test_repeated_convolution_order():
    """Each repeated convolution must be followed by batch normalization and ReLU."""
    inputs = tf.keras.Input(shape=(4, 4, 4, 1))
    outputs = networks.conv_block(inputs, 2, number=2)
    model = tf.keras.Model(inputs, outputs)
    layer_types = [type(layer) for layer in model.layers[1:]]
    expected = [tf.keras.layers.Conv3D, tf.keras.layers.BatchNormalization, tf.keras.layers.Activation] * 2
    assert layer_types == expected


@pytest.mark.parametrize("block", ["bottleneck", "decoder"])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("rate", [0.0, 0.5])
def test_spatial_dropout_is_connected_after_block(block, residual, rate):
    """Nonzero rates add channel dropout; inference preserves its input exactly."""
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(2, 2, 2, 4))
    model_inputs = inputs
    if block == "bottleneck":
        outputs = networks.bottleneck_block(inputs, 32, residual=residual, dropout=rate)
    else:
        skip = tf.keras.Input(shape=(4, 4, 4, 32))
        model_inputs = [inputs, skip]
        outputs = networks.decoder_block(inputs, skip, 32, residual=residual, dropout=rate)

    model = tf.keras.Model(model_inputs, outputs)
    dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.SpatialDropout3D)]
    if rate == 0:
        assert not dropout_layers
        return

    assert len(dropout_layers) == 1
    dropout = dropout_layers[0]
    assert model.layers[-1] is dropout
    assert dropout.rate == rate

    shape = tuple(2 if dimension is None else dimension for dimension in dropout.input_shape)
    values = tf.ones(shape)
    inference = dropout(values, training=False).numpy()
    np.testing.assert_array_equal(inference, values.numpy())

    tf.random.set_seed(42)
    training = dropout(values, training=True).numpy()
    assert set(np.unique(training)) == {0.0, 2.0}
    # Each channel shares one dropout value across all its spatial positions.
    channel_values = training[:, :1, :1, :1, :]
    expected = np.broadcast_to(channel_values, training.shape)
    np.testing.assert_array_equal(training, expected)


@pytest.mark.parametrize("residual", [False, True])
def test_multiscale_residual_selection(residual):
    """Turning residuals off keeps pooled features but removes the shortcut."""
    inputs = tf.keras.Input(shape=(8, 8, 8, 1))
    features, pooled = networks.multiscale_block(inputs, 4, (2, 2, 2), residual=residual)
    model = tf.keras.Model(inputs, [features, pooled])
    additions = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Add)]
    assert len(additions) == int(residual)
    assert model.output_shape == [(None, 4, 4, 4, 4), (None, 4, 4, 4, 1)]


def test_multitask_forward_outputs_remain_valid():
    """Smoke-check all three heads numerically without training a model."""
    tf.keras.backend.clear_session()
    model = networks.mt_r_net_3d((16, 16, 16), 1, 3, [2, 4, 8, 16, 32], with_classifier=True)
    values = tf.ones((1, 16, 16, 16, 1))
    regions, brain, condition = model(values, training=False)
    for output in (regions, brain, condition):
        assert np.isfinite(output.numpy()).all()
        assert np.all(output.numpy() >= 0) and np.all(output.numpy() <= 1)
    np.testing.assert_allclose(tf.reduce_sum(regions, axis=-1).numpy(), 1, atol=1e-6)
