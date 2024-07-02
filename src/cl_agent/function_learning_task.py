from typing import Tuple

import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx

from utils import tree_replace


class FunctionLearningTaskState(eqx.Module):
    rng: Array

    # Params
    n_inputs: int = eqx.field(static=True)
    n_distractors: int = eqx.field(static=True)
    change_freq: int = eqx.field(static=True)
    prediction_delay: int = eqx.field(static=True)

    # State vars
    step_idx: Array
    input_weights: Array
    input_history: Array
    output_history: Array


def gen_weights(rng: Array, n: int) -> Array:
    """Generates a random array of 1s and -1s."""
    return jax.random.choice(rng, jnp.array([-1.0, 1.0]), (n,))


def init_function_learning_task(
        rng: Array,
        n_inputs: int = 20,
        n_distractors: int = 15,
        change_freq: int = 20,
        prediction_delay: int = 0,
    ) -> FunctionLearningTaskState:
    """Initializes the function learning task state.
    
    Args:
        rng (Array): RNG key
        n_inputs (int): Total number of inputs.
        n_distractors (int): Number of distractors / irrelevant inputs.
        prediction_delay (int): Prediction delay.

    Returns:
        FunctionLearningTaskState: The initialized function learning task state.
    """
    weights_key, rng = jax.random.split(rng)
    input_weights = jnp.concat([
        gen_weights(weights_key, n_inputs - n_distractors),
        jnp.zeros(n_distractors, dtype=jnp.float32),
    ])

    return FunctionLearningTaskState(
        rng = rng,
        n_inputs = n_inputs,
        n_distractors = n_distractors,
        change_freq = change_freq,
        prediction_delay = prediction_delay,
        step_idx = jnp.array(0),
        input_weights = input_weights,
        input_history = jnp.zeros((prediction_delay + 1, n_inputs,)),
        output_history = jnp.zeros((prediction_delay + 1,)),
    )


def step_function_learning_task(state: FunctionLearningTaskState) -> Tuple[FunctionLearningTaskState, Tuple[Array]]:
    """Returns the next state and the inputs and outputs for the function learning task.

    There are n_inputs inputs, and the target output is the sum of the weights times the inputs.
    The weights are randomly generated arrays of 1s and -1s, and one changes every change_freq steps.
    The weights for the n_distractors are always 0.
    
    Args:
        state (FunctionLearningTaskState): The current state of the function learning task.

    Returns:
        Tuple[FunctionLearningTaskState, Tuple[Array]]: The updated state and the inputs and outputs.
    """
    input_key, weight_key, rng = jax.random.split(state.rng, 3)

    # Change the weights if necessary
    n_important_inputs = state.n_inputs - state.n_distractors
    weight_change_idx = jax.random.randint(weight_key, (1,), 0, n_important_inputs)
    input_weights = jax.lax.cond(
        state.step_idx % state.change_freq == 0,
        lambda: state.input_weights.at[weight_change_idx].set(-state.input_weights[weight_change_idx]),
        lambda: state.input_weights,
    )

    # Generate inputs and outputs
    inputs = jax.random.normal(input_key, (state.n_inputs,))
    input_history = jnp.concatenate([inputs[None], state.input_history[:-1]])

    outputs = jnp.sum(inputs * input_weights)
    output_history = jnp.concatenate([outputs[None], state.output_history[:-1]])

    state = tree_replace(
        state,
        rng = rng,
        step_idx = state.step_idx + 1,
        input_history = input_history,
        output_history = output_history,
        input_weights = input_weights,
    )

    return state, (input_history.flatten(), output_history[-1])


if __name__ == '__main__':
    state = init_function_learning_task(jax.random.PRNGKey(0), n_inputs=3, n_distractors=1, change_freq=5)

    step_fn = jax.jit(step_function_learning_task)

    for _ in range(10):
        state, (x, y) = step_fn(state)
        print(x, y)