from typing import Callable, Sequence, Tuple

import equinox as eqx
import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import PyTree
import optax

from cl_agent.models import LSTMState, 
from swift_td import SwiftTDState, swift_td_step


class TrainState(eqx.Module):
    swift_td_state: SwiftTDState
    opt_state: optax.OptState
    obs_history: Array
    cumulant_history: Array
    train_step: int
    tx_update_fn: Callable = eqx.field(static=True)
    tbptt_window: int = eqx.field(static=True)
    # How frequently to update weights connected into hidden layers
    feature_update_freq: int = eqx.field(static=True)

    def __init__(
            self,
            swift_td_state: SwiftTDState,
            opt_state: optax.OptState,
            tx_update_fn: Callable,
            obs_shape: Sequence,
            tbptt_window: int = 4,
            feature_update_freq: int = 4,
        ):
        self.swift_td_state = swift_td_state
        self.opt_state = opt_state
        self.tx_update_fn = tx_update_fn
        self.train_step = 0
        self.tbptt_window = tbptt_window
        self.feature_update_freq = feature_update_freq
    


def train_gvf_step(
        train_state: TrainState,
        rnn_state: LSTMState,
        model: eqx.Module,
        obs: Array,
        cumulant: float,
    ) -> Tuple[TrainState, PyTree, LSTMState, eqx.Module]:
    """Perform a single training step.
    
    Args:
        train_state: The current training state.
        rnn_state: The current RNN state.
        model: The current model.
        obs: Observation from the environment.
        cumulant: Cumulant from the environment (possibly the reward).

    Returns:
        The new training state, RNN state, and model.
    """

    pass

    # Update hidden layer weights if necessary


    # Update output layer weights
    swift_td_step(train_state.swift_td_state, obs, cumulant)

    # Increment train step

    # Return updated state, rnn_state, and model