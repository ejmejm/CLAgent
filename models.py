from typing import Tuple, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp


class SupervisedLSTMModel(nn.Module):
    vocab_size: int
    lstm_input_sizes: Tuple[int] = (128, 512)

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.lstm_input_sizes[0])
        self.lstm1 = nn.OptimizedLSTMCell(128)
        self.dense1 = nn.Dense(self.lstm_input_sizes[1])
        self.lstm2 = nn.OptimizedLSTMCell(128)
        self.dense2 = nn.Dense(512)
        self.out_layer = nn.Dense(self.vocab_size, use_bias=False)

    def init_rnn_state(self):
        return (
            self.lstm1.initialize_carry(jax.random.PRNGKey(0), (self.lstm_input_sizes[0],)),
            self.lstm2.initialize_carry(jax.random.PRNGKey(0), (self.lstm_input_sizes[1],)),
        )

    @nn.remat    
    def __call__(self, x: jnp.ndarray, rnn_state: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        intermediates = []
        z = self.embedding(x)
        intermediates.append(z)
        
        if rnn_state is None:
            rnn_state = self.init_rnn_state()
        
        rnn_state_1, z = self.lstm1(rnn_state[0], z)
        z = nn.relu(z)
        intermediates.append(z)
        
        z = self.dense1(z)
        z = nn.relu(z)
        intermediates.append(z)

        rnn_state_2, z = self.lstm2(rnn_state[1], z)
        z = nn.relu(z)
        intermediates.append(z)

        z = self.dense2(z)
        z = nn.relu(z)
        intermediates.append(z)

        out = self.out_layer(z)
        intermediates.append(out)
        return (rnn_state_1, rnn_state_2), z, out

    @staticmethod
    def forward_seq(model, params, rnn_state, xs):
        def step(rnn_state, x):
            rnn_state, z, y = model.apply(params, x, rnn_state)
            return rnn_state, (z, y)
        return jax.lax.scan(step, rnn_state, xs)