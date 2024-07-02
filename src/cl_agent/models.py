import math
from typing import List, Tuple, Optional, Sequence, Union

import equinox as eqx
from equinox import nn
import jax
from jax import Array
import jax.numpy as jnp


LSTMState = Tuple[Tuple[Array]]


class SupervisedLSTMModel(eqx.Module):
    vocab_size: int = eqx.field(static=True)
    embedding: nn.Embedding
    lstm_layers: List[nn.LSTMCell]
    linear_layers: List[nn.Linear]

    def __init__(self, key: jax.Array, vocab_size: int, lstm_input_sizes: Tuple[int] = (128, 128)):
        gen_keys = jax.random.split(key, 6)

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, lstm_input_sizes[0], key=gen_keys[0])
        self.lstm_layers = []
        self.linear_layers = []

        self.lstm_layers.append(nn.LSTMCell(lstm_input_sizes[0], 128, key=gen_keys[1]))
        self.linear_layers.append(nn.Linear(128, lstm_input_sizes[1], key=gen_keys[2]))
        self.lstm_layers.append(nn.LSTMCell(lstm_input_sizes[1], 128, key=gen_keys[3]))
        self.linear_layers.append(nn.Linear(128, 256, key=gen_keys[4]))
        self.linear_layers.append(nn.Linear(256, self.vocab_size, use_bias=False, key=gen_keys[5]))

    def init_rnn_state(self):
        return (
            (jnp.zeros(self.lstm_layers[0].hidden_size), jnp.zeros(self.lstm_layers[0].hidden_size)),
            (jnp.zeros(self.lstm_layers[1].hidden_size), jnp.zeros(self.lstm_layers[1].hidden_size)),
        )

    @jax.remat
    def __call__(self, x: jnp.ndarray, rnn_state: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        intermediates = []
        z = self.embedding(x)
        intermediates.append(z)
        
        if rnn_state is None:
            rnn_state = self.init_rnn_state()
        
        rnn_state_1 = self.lstm_layers[0](z, rnn_state[0])
        z = rnn_state_1[0]
        z = jax.nn.relu(z)
        intermediates.append(z)
        
        z = self.linear_layers[0](z)
        z = jax.nn.relu(z)
        intermediates.append(z)

        rnn_state_2 = self.lstm_layers[1](z, rnn_state[1])
        z = rnn_state_2[0]
        z = jax.nn.relu(z)
        intermediates.append(z)

        z = self.linear_layers[1](z)
        z = jax.nn.relu(z)
        intermediates.append(z)

        out = self.linear_layers[2](z)
        intermediates.append(out)

        return (rnn_state_1, rnn_state_2), intermediates, out

    def forward_sequence(self, rnn_state, xs):
        def step(rnn_state, x):
            rnn_state, z, y = self.__call__(x, rnn_state)
            return rnn_state, (z, y)
        state, (zs, ys) = jax.lax.scan(step, rnn_state, xs)
        return state, zs, ys
    

class FeatureExtractor(eqx.Module):
    obs_dim: int = eqx.field(static=True)
    layer_sizes: Sequence[int] = eqx.field(static=True)
    recurrent_layer_indices: List[int] = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    layers: List[eqx.Module]

    def __init__(
            self,
            key: Array,
            obs_dim: Union[int, Sequence[int]],
            layer_sizes: Sequence[int],
            output_dim: int,
            recurrent_layer_indices: List[int],
        ):
        if isinstance(obs_dim, int):
            obs_dim = (obs_dim,)

        self.obs_dim = math.prod(obs_dim)
        self.output_dim = output_dim
        self.recurrent_layer_indices = recurrent_layer_indices
        self.n_layers = len(layer_sizes) + 1

        gen_keys = jax.random.split(key, len(layer_sizes) - 1)
        self.layer_sizes = [self.obs_dim] + layer_sizes + [output_dim]
        
        self.layers = []
        for i in range(1, len(self.layer_sizes)):
            if i - 1 in self.recurrent_layer_indices:
                self.layers.append(nn.LSTMCell(self.layer_sizes[i - 1], self.layer_sizes[i], key=gen_keys[i - 1]))
            else:
                self.layers.append(nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i], key=gen_keys[i - 1]))

    def init_rnn_state(self):
        return tuple([
            (jnp.zeros(self.layer_sizes[i]),
             jnp.zeros(self.layer_sizes[i]))
            for i in self.recurrent_layer_indices
        ])
    
    @jax.remat
    def __call__(self, x: Array, rnn_state: Optional[LSTMState] = None) -> Array:
        z = x.flatten()

        recurrent_layer_idx = 0
        new_rnn_state = []
        intermediates = []
        for i in range(self.n_layers):
            layer = self.layers[i]

            if i in self.recurrent_layer_indices:
                z = layer(z, rnn_state[recurrent_layer_idx])
                new_rnn_state.append(z)
                z = z[0]
                recurrent_layer_idx += 1
            else:
                z = layer(z)
                print(z.shape)

            if i < self.n_layers - 1:
                z = jax.nn.relu(z)

            intermediates.append(z)

        return tuple(new_rnn_state), intermediates, z


class RecurrentGVFModel(eqx.Module):
    obs_dim: int = eqx.field(static=True)
    lstm_layers: List[nn.LSTMCell]
    linear_layers: List[nn.Linear]
    output_layer: nn.Linear

    def __init__(self, key: jax.Array, vocab_size: int, lstm_input_sizes: Tuple[int] = (128, 128)):
        gen_keys = jax.random.split(key, 6)

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, lstm_input_sizes[0], key=gen_keys[0])
        self.lstm_layers = []
        self.linear_layers = []

        self.lstm_layers.append(nn.LSTMCell(lstm_input_sizes[0], 128, key=gen_keys[1]))
        self.linear_layers.append(nn.Linear(128, lstm_input_sizes[1], key=gen_keys[2]))
        self.lstm_layers.append(nn.LSTMCell(lstm_input_sizes[1], 128, key=gen_keys[3]))
        self.linear_layers.append(nn.Linear(128, 256, key=gen_keys[4]))
        self.output_layer = nn.Linear(256, 1, use_bias=False, key=gen_keys[5])

    def init_rnn_state(self):
        return (
            (jnp.zeros(self.lstm_layers[0].hidden_size), jnp.zeros(self.lstm_layers[0].hidden_size)),
            (jnp.zeros(self.lstm_layers[1].hidden_size), jnp.zeros(self.lstm_layers[1].hidden_size)),
        )

    @jax.remat
    def __call__(self, x: jnp.ndarray, rnn_state: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        intermediates = []
        z = self.embedding(x)
        intermediates.append(z)
        
        if rnn_state is None:
            rnn_state = self.init_rnn_state()
        
        rnn_state_1 = self.lstm_layers[0](z, rnn_state[0])
        z = rnn_state_1[0]
        z = jax.nn.relu(z)
        intermediates.append(z)
        
        z = self.linear_layers[0](z)
        z = jax.nn.relu(z)
        intermediates.append(z)

        rnn_state_2 = self.lstm_layers[1](z, rnn_state[1])
        z = rnn_state_2[0]
        z = jax.nn.relu(z)
        intermediates.append(z)

        z = self.linear_layers[1](z)
        z = jax.nn.relu(z)
        intermediates.append(z)

        out = self.linear_layers[2](z)
        intermediates.append(out)

        return (rnn_state_1, rnn_state_2), intermediates, out

    def forward_sequence(self, rnn_state, xs):
        def step(rnn_state, x):
            rnn_state, z, y = self.__call__(x, rnn_state)
            return rnn_state, (z, y)
        state, (zs, ys) = jax.lax.scan(step, rnn_state, xs)
        return state, zs, ys
    

if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)

    model_key, rng = jax.random.split(rng)
    model = SupervisedLSTMModel(model_key, 10)
    print(model)

    x = jnp.zeros((20,), dtype=jnp.int32)
    rnn_state = model.init_rnn_state()

    # Test single forward pass
    forward = jax.jit(model.__call__)
    print('Input shape:', x[0].shape)
    state, _, y = forward(x[0], rnn_state)
    print(f'Output shape: {y.shape} | Hidden state 1 shape: {state[0][0].shape} | Cell state 1 shape: {state[1][0].shape}')

    # Test sequence forward pass
    forward_sequence = eqx.filter_jit(model.forward_sequence)
    state, zs, ys = forward_sequence(rnn_state, x)
    print(f'Output shape: {ys.shape} | Hidden state 1 shape: {state[0][0].shape} | Cell state 1 shape: {state[1][0].shape}')

    # Test FeatureExtractor
    model_key, rng = jax.random.split(rng)
    feature_extractor = FeatureExtractor(model_key, 10, [32, 32], 64, [1])
    print(feature_extractor)
    
    x = jnp.zeros((10,), dtype=jnp.float32)
    rnn_state = feature_extractor.init_rnn_state()
    forward = feature_extractor.__call__ # jax.jit(feature_extractor.__call__)
    state, intermediates, y = forward(x, rnn_state)
    print(f'Output shape: {y.shape} | Hidden state 1 shape: {state[0][0].shape} | Cell state 1 shape: {state[0][1].shape}')
    print(f'Intermediates: {[i.shape for i in intermediates]}')