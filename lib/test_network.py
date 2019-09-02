import pytest
import torch
from lib.network import LinearNet


def check_linear_layer(layer, in_features, out_features, has_bias=True):
    assert layer.in_features == in_features
    assert layer.out_features == out_features
    if has_bias:
        assert layer.bias is not None
    else:
        assert layer.bias is None


def test_constructor_default_layers():
    net = LinearNet()
    check_linear_layer(net.layer1, 2, 4)
    check_linear_layer(net.layer2, 4, 8)
    check_linear_layer(net.layer3, 8, 4)
    check_linear_layer(net.layer4, 4, 1)


def test_constructor_custom_layers():
    net = LinearNet(nodes=[1000, 8, 16, 9])
    check_linear_layer(net.layer1, 1000, 8)
    check_linear_layer(net.layer2, 8, 16)
    check_linear_layer(net.layer3, 16, 9)
    with pytest.raises(AttributeError):
        net.layer4
