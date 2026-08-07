"""Shared activation functions for experiment comparisons.

Use these helpers to swap nonlinearities from the training command line
without changing model code.
"""

from __future__ import annotations

from typing import Final

import torch as th
from torch import nn


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""

    def forward(self, input_tensor: th.Tensor) -> th.Tensor:
        return input_tensor * th.sigmoid(input_tensor)


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""

    def forward(self, input_tensor: th.Tensor) -> th.Tensor:
        return input_tensor * th.tanh(th.nn.functional.softplus(input_tensor))


class Identity(nn.Module):
    """No-op activation for control experiments."""

    def forward(self, input_tensor: th.Tensor) -> th.Tensor:
        return input_tensor


ACTIVATION_FUNCTIONS: Final[dict[str, type[nn.Module]]] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": Swish,
    "mish": Mish,
    "identity": Identity,
}


def available_activation_names() -> list[str]:
    return sorted(ACTIVATION_FUNCTIONS)


def get_activation_fn(name: str) -> type[nn.Module]:
    key = name.strip().lower()
    if key not in ACTIVATION_FUNCTIONS:
        valid = ", ".join(available_activation_names())
        raise ValueError(f"Unknown activation '{name}'. Choose one of: {valid}")
    return ACTIVATION_FUNCTIONS[key]
