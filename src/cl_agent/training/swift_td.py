from typing import Tuple

import equinox as eqx
import jax
from jax import Array
import jax.numpy as jnp

from cl_agent.utils import tree_replace


class SwiftTDState(eqx.Module):
    # Static params
    n_features: int = eqx.field(static=True) # Number of input features
    meta_lr: float = eqx.field(static=True) # Meta learning rate

    epsilon: float = eqx.field(static=True) # LR decay factor
    eta: float = eqx.field(static=True) # Max learning rate
    trace_decay: float = eqx.field(static=True) # Lambda trace decay
    gamma: float = eqx.field(static=True) # Discount factor

    # State vars
    beta: Array # Learning rate exponent
    h_old: Array
    h_temp: Array
    z_delta: Array
    p: Array # Eligibility trace of lr exponent
    h: Array
    z: Array # Eligibility trace of weights
    z_bar: Array
    V_delta: Array
    V_old: Array

    def __init__(
            self,
            n_features,
            lr_init: float = 1e-7,
            meta_lr: float = 1e-3,
            epsilon: float = 0.9,
            eta: float = 0.5,
            trace_decay: float = 0.95,
            gamma: float = 0.1,
        ):
        self.n_features = n_features
        self.beta = jnp.log(lr_init) * jnp.ones(self.n_features)
        self.meta_lr = meta_lr
        self.epsilon = epsilon
        self.eta = eta
        self.trace_decay = trace_decay
        self.gamma = gamma

        self.h_old, self.h_temp, self.z_delta, self.p, self.h, self.z, self.z_bar = [jnp.zeros(self.n_features) for _ in range(7)]
        self.V_delta, self.V_old = [jnp.array(0.0) for _ in range(2)]


def swift_td_step(
        state: SwiftTDState,
        weights: Array,
        features: Array,
        cumulant: float,
    ) -> Tuple[SwiftTDState, Array, float]:
    """SwiftTD update step.
    
    Args:
        state (SwiftTDState): Current state of the Swift-TD algorithm.
        weights (Array): Current weights of the model. Must have elements of shape (n_features,).
        features (Array): Input features.
        cumulant (Array): Scalar cumulant signal.

    Returns:
        Tuple[SwiftTDState, Array, float]: Updated state, updated weights, and TD error.
    """
    orig_weight_shape = weights.shape
    weights = weights.flatten()
    V = jnp.dot(weights, features)
    delta = cumulant + state.gamma * V - state.V_old

    # Weight and lr updates
    out = {}
    delta_w = delta * state.z - state.z_delta * state.V_delta
    delta_w = jnp.where(state.z == 0, jnp.zeros_like(features), delta_w)
    weights = jnp.where(state.z == 0, weights, weights + delta_w) # Weight update

    out['beta'] = state.beta + state.meta_lr / (jnp.exp(state.beta) + 1e-8) # Meta learning rate update
    out['beta'] = jnp.minimum(out['beta'], jnp.log(state.eta)) # Clip learning rate
    out['h_old'] = state.h
    out['h'] = state.h_temp
    out['h_temp'] = out['h'] + delta * state.z_bar - state.z_delta * state.V_delta
    out['z_delta'] = jnp.zeros_like(state.z_delta)

    # Decay traces
    out['z'] = state.gamma * state.trace_decay * state.z
    out['p'] = state.gamma * state.trace_decay * state.p
    out['z_bar'] = state.gamma * state.trace_decay * state.z_bar

    # Replace state variables with out values only where z != 0
    state = tree_replace(
        state,
        **{k: jnp.where(state.z == 0, getattr(state, k), v) for k, v in out.items()}
    )

    state = tree_replace(state, V_delta=jnp.array(0.0))
    lr = jnp.exp(state.beta)
    E = jnp.maximum(jnp.array(state.eta), jnp.dot(lr, features ** 2)) # Rate of learning
    T = jnp.dot(state.z, features)
    state = tree_replace(state, V_delta=state.V_delta + jnp.dot(delta_w, features)) # Minor error because delta_w may not be defined

    # Eligibility trace updates
    out = {}
    out['z_delta'] = state.eta / E * jnp.exp(state.beta) * features
    out['z'] = state.z + out['z_delta'] * (1 - T) # Update weight eligibility trace
    out['p'] = state.p + features * state.h # Update lr eligibility trace
    out['z_bar'] = state.z_bar + out['z_delta'] * (1 - T - features * state.z_bar)
    out['h_temp'] = state.h_temp - state.h_old * features * (out['z'] - out['z_delta']) \
        - state.h * out['z_delta'] * features
    
    # Conditionally decay lr
    out['beta'] = jax.lax.cond(
        E <= state.eta,
        lambda beta: beta,
        lambda beta: beta + jnp.abs(features) * jnp.log(state.epsilon),
        state.beta,
    )
    out['h_temp'], out['h'], out['z_bar'] = jax.lax.cond(
        E <= state.eta,
        lambda h_temp, h, z_bar: (h_temp, h, z_bar),
        lambda h_temp, h, z_bar: (jnp.zeros_like(h_temp), jnp.zeros_like(h), jnp.zeros_like(z_bar)),
        out['h_temp'], state.h, out['z_bar'],
    )

    # Replace state variables with out values only where features != 0
    state = tree_replace(
        state,
        **{k: jnp.where(features == 0, getattr(state, k), v) for k, v in out.items()}
    )

    state = tree_replace(state, V_old=V)

    return state, weights.reshape(orig_weight_shape), delta
