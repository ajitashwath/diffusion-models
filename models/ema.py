"""
models/ema.py
=============
Exponential Moving Average (EMA) of model weights.

Maintaining an EMA of model weights during training is a simple but effective
technique to produce a smoother, more stable model for inference.  The EMA
weights are typically used only for evaluation / sampling and are *not* used
to compute gradients.

Reference: https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html
"""

import copy
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Optional

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average of a model's parameters.

    Maintains a *shadow* copy of each parameter p:

        shadow = decay * shadow + (1 - decay) * p

    The shadow copy is updated after every optimizer step via :meth:`update`.
    For evaluation, swap in the EMA weights using :meth:`copy_to` or the
    context manager (:meth:`__enter__` / :meth:`__exit__`).

    Parameters
    ----------
    model : nn.Module
        The model whose parameters should be tracked.
    decay : float
        EMA decay rate.  Common values: 0.999 – 0.9999.  Higher values
        result in a smoother (but more inertia-laden) moving average.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")

        self.decay = decay
        self._shadow_params: List[torch.Tensor] = [
            p.clone().detach() for p in model.parameters()
        ]
        # Storage for the context-manager's saved original parameters
        self._stored_params: Optional[List[torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Core EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow parameters with a moving average step.

        Must be called **after** each ``optimizer.step()`` to keep the EMA
        weights in sync with the latest model weights.

        Parameters
        ----------
        model : nn.Module
            The model being trained (live, gradient-updated parameters).
        """
        decay = self.decay
        one_minus_decay = 1.0 - decay
        for shadow, param in zip(self._shadow_params, model.parameters()):
            if param.requires_grad:
                shadow.mul_(decay).add_(param.data, alpha=one_minus_decay)
            else:
                # Non-learnable buffers (e.g. running stats): copy directly
                shadow.copy_(param.data)

    # ------------------------------------------------------------------
    # Copy / restore helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy shadow (EMA) parameters into ``model``.

        Useful for evaluation: call ``copy_to`` before running inference so
        the model uses the smoother EMA weights.

        Parameters
        ----------
        model : nn.Module
            Destination model whose parameters will be overwritten.
        """
        for shadow, param in zip(self._shadow_params, model.parameters()):
            param.data.copy_(shadow)

    def store(self, parameters: Iterable[nn.Parameter]) -> None:
        """
        Save a snapshot of ``parameters`` so they can be restored later.

        Typically used internally by the context manager, but can also be
        called manually to checkpoint the current live weights before swapping
        in EMA weights.

        Parameters
        ----------
        parameters : Iterable[nn.Parameter]
            The parameters to save (e.g. ``model.parameters()``).
        """
        self._stored_params = [p.clone().detach() for p in parameters]

    @torch.no_grad()
    def restore(self, parameters: Iterable[nn.Parameter]) -> None:
        """
        Restore parameters previously saved by :meth:`store`.

        Parameters
        ----------
        parameters : Iterable[nn.Parameter]
            The parameters to restore into (e.g. ``model.parameters()``).

        Raises
        ------
        RuntimeError
            If :meth:`store` has not been called before :meth:`restore`.
        """
        if self._stored_params is None:
            raise RuntimeError(
                "EMA.restore() called before EMA.store(). "
                "Make sure to call store() first."
            )
        for stored, param in zip(self._stored_params, parameters):
            param.data.copy_(stored)
        self._stored_params = None  # clear after restore

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "EMA":
        """
        Enter the EMA context: save current model weights and apply EMA weights.

        Usage::

            with ema:
                outputs = model(inputs)  # model now uses EMA weights

        .. warning::
            The ``model`` must be provided to ``copy_to`` separately.  Use
            :meth:`average_parameters` context manager for an all-in-one
            approach.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the EMA context: do nothing (restoration must be done manually)."""
        return False  # do not suppress exceptions

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """
        Context manager that temporarily swaps ``model``'s parameters for the
        EMA shadow parameters, then restores the original parameters on exit.

        Parameters
        ----------
        model : nn.Module
            Model to swap parameters for.

        Example
        -------
        ::

            with ema.average_parameters(model):
                samples = model(noise)  # uses EMA weights
            # Back to training weights here
        """
        self.store(model.parameters())
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model.parameters())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a state dict containing shadow parameters for checkpointing."""
        return {
            "decay": self.decay,
            "shadow_params": [p.cpu() for p in self._shadow_params],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load shadow parameters from a previously saved state dict."""
        self.decay = state_dict["decay"]
        self._shadow_params = [
            p.clone().detach() for p in state_dict["shadow_params"]
        ]

    def __repr__(self) -> str:
        return f"EMA(decay={self.decay}, num_params={len(self._shadow_params)})"
