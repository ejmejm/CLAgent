from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
from jax import Array
import jax.numpy as jnp
import optax

from models import LSTMState


def train_on_sequence(
        model: eqx.Module,
        opt_state: optax.OptState,
        tx_update_fn: Callable,
        rnn_state: LSTMState,
        sequence: Array,
        tbptt_window: int = 4,
    ):
    def loss_fn(model: eqx.Module, rnn_states: LSTMState, input_tokens: Array, target_tokens: Array, loss_mask: Array):
        rnn_states, _, pred_ids = model.forward_sequence(rnn_states, input_tokens)
        loss = optax.softmax_cross_entropy_with_integer_labels(pred_ids, target_tokens) * loss_mask
        n_targets = jnp.sum(loss_mask)
        loss = jax.lax.cond(n_targets > 0, lambda x: jnp.sum(x) / n_targets, lambda _: 0.0, loss)
        return loss, rnn_states
    
    value_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    def update_fn(
            state: Tuple[optax.OptState, eqx.Module, LSTMState],
            sequence: Dict[str, Array],
        ):
        opt_state, model, init_rnn_state = state
        (loss, new_rnn_state), grads = value_grad_fn(
            model, init_rnn_state, sequence['input_ids'], sequence['target_ids'], sequence['loss_mask'])
        updates, new_opt_state = tx_update_fn(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return (new_opt_state, new_model, new_rnn_state), loss
    
    # Pad the sequence
    remainder = sequence['input_ids'].shape[0] % tbptt_window
    n_pad = (tbptt_window - remainder) % tbptt_window
    sequence = jax.tree.map(
        lambda x: jnp.pad(x, (0, n_pad), mode='constant'),
        sequence,
    )

    # Split the input sequence into subsequences of length tbptt_window
    subsequences = jax.tree.map(
        lambda x: x.reshape((-1, tbptt_window)),
        sequence,
    )

    (new_opt_state, new_model, final_rnn_state), loss = \
        jax.lax.scan(update_fn, (opt_state, model, rnn_state), subsequences)

    return new_opt_state, new_model, final_rnn_state, jnp.mean(loss)


# def train_on_sequences(opt_state, variables, sequences, init_rnn_state_fn, apply_fn, tx_update_fn, tbptt_window=4):
    
#     def loss_fn(params, rnn_states, input_tokens, target_tokens, loss_mask):
#         input_tokens = jnp.swapaxes(input_tokens, 0, 1)
#         rnn_states, pred_ids = apply_fn({'params': params}, rnn_states, input_tokens)
#         pred_ids = jnp.swapaxes(pred_ids, 0, 1)
#         loss = optax.softmax_cross_entropy_with_integer_labels(pred_ids, target_tokens) * loss_mask
#         loss = jnp.sum(loss) / jnp.sum(loss_mask)
#         return loss, rnn_states
    
#     batch_size = sequences['input_ids'].shape[0]
#     params = variables.pop('params')
#     value_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

#     def update_fn(state, sequences):
#         opt_state, params, init_rnn_states = state
#         (loss, new_rnn_state), grads = value_grad_fn(
#             params, init_rnn_states, sequences['input_ids'], sequences['target_ids'], sequences['loss_mask'])
#         updates, new_opt_state = tx_update_fn(grads, opt_state, params)
#         new_params = optax.apply_updates(params, updates)
#         return (new_opt_state, new_params, new_rnn_state), loss

#     # Pad the sequence
#     n_pad = tbptt_window - (sequences['input_ids'].shape[1] % tbptt_window)
#     sequences = jax.tree.map(
#         lambda x: jnp.pad(x, ((0, 0), (0, n_pad)), mode='constant'),
#         sequences,
#     )

#     # Split the input sequence into subsequences of length tbptt_window
#     subsequences = jax.tree.map(
#         lambda x: x.reshape((batch_size, -1, tbptt_window)),
#         sequences,
#     )

#     init_rnn_states = init_rnn_state_fn() # jax.vmap(lambda _: init_rnn_state_fn())(jnp.ones(batch_size))
#     def repeat_state(arr):
#         arr = jnp.expand_dims(arr, 0)
#         return jnp.repeat(arr, batch_size, axis=0)
#     init_rnn_states = jax.tree.map(repeat_state, init_rnn_states)

#     # (new_opt_state, new_params, final_rnn_state), loss = \
#     #     jax.lax.scan(update_fn, (opt_state, params, init_rnn_state), subsequences)
#     (new_opt_state, new_params, final_rnn_state), loss = update_fn((opt_state, params, init_rnn_states), sequences)

#     new_variables = {**variables, 'params': new_params}
#     return new_opt_state, new_variables, final_rnn_state, loss