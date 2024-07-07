import argparse
from typing import Dict, List

import gymnax
import jax
import numpy as np
from tabulate import tabulate

from cl_agent.envs import EquinoxEnv, GymnaxContinualEnvWrapper, PenalizeTerminationWrapper
from cl_agent.models import FeatureExtractor, ActorCriticModel
from cl_agent.training.training import TrainState, train_loop
from cl_agent.training.swift_td import SwiftTDState
from optax import adam


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Actor-Critic model with given parameters.")

    parser.add_argument(
        '--env_name', type=str, required=True, help="Name of the environment")
    parser.add_argument(
        '--seed', type=int, default=0, help="Random seed")
    parser.add_argument(
        '--gamma', type=float, default=0.99, help="Discount factor for future rewards")
    
    # Train arguments
    parser.add_argument(
        '--train_steps', type=int, default=1000000, help="Number of training steps")
    parser.add_argument(
        '--log_freq', type=int, default=10000, help="Number of steps between logging metrics")
    
    # Feature extractor model params
    parser.add_argument(
        '--fe_layer_sizes', type=int, nargs='+', default=[64, 64],
        help="Sizes of the layers in the feature extractor network",
    )
    parser.add_argument(
        '--fe_latent_dim', type=int, default=64,
        help="Size of the latent dimension of the feature extractor",
    )
    parser.add_argument(
        '--fe_recurrent_layer_indices', type=int, nargs='*', default=[1],
        help="Indices of the feature extractor layers that are recurrent",
    )

    # Actor-critic model params
    parser.add_argument(
        '--actor_layer_sizes', type=int, nargs='+', default=[64, 32],
        help="Sizes of the layers in the actor network"
    )
    parser.add_argument(
        '--critic_layer_sizes', type=int, nargs='+', default=[64, 32],
        help="Sizes of the layers in the critic network"
    )

    # Backprop training params
    parser.add_argument(
        '--learning_rate', type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument(
        '--feature_update_freq', type=int, default=1, help="Frequency of feature updates")
    parser.add_argument(
        '--tbptt_window', type=int, default=4,
        help="Size of truncated backprop through time window (1 is the minimum)",
    )

    # SwiftTD parameters
    parser.add_argument(
        '--no_swift_td', dest='use_swift_td', action='store_false', help="Whether to use SwiftTD")
    parser.add_argument(
        '--std_lr_init', type=float, default=1e-7, help="Initial learning rate")
    parser.add_argument(
        '--std_meta_lr', type=float, default=1e-3, help="Meta learning rate")
    parser.add_argument(
        '--std_epsilon', type=float, default=0.9, help="LR decay factor")
    parser.add_argument(
        '--std_eta', type=float, default=0.5, help="Max learning rate")
    parser.add_argument(
        '--std_trace_decay', type=float, default=0.95, help="Lambda trace decay")
    parser.add_argument(
        '--std_gamma', type=float, default=0.1, help="Discount factor")

    args = parser.parse_args()
    return args


def print_metrics(timestep: int, metrics: Dict[str, List[float]]):
    # Function to calculate mean and 95% confidence interval
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        mean, se = np.mean(a), np.std(a) / np.sqrt(n)
        h = se * 1.96  # 1.96 corresponds to 95% confidence
        return mean, h
    
    # Prepare data for tabulate
    table_data = []
    for metric_name, values in metrics.items():
        mean, ci = mean_confidence_interval(values)
        table_data.append([metric_name, f"{mean:.5f}", f"±{ci:.5f}"])
    
    # Create table
    table = tabulate(table_data, headers=["Metric", "Mean", "95% CI"], tablefmt="pretty")
    
    # Print the timestep and table
    print(f"Timestep: {timestep}")
    print(table)


def train(args: argparse.Namespace):
    # Set the random seed
    rng = jax.random.PRNGKey(args.seed)
    env_key, fe_key, ac_key, train_key, rng = jax.random.split(rng, 5)


    # Init the environment
    env, env_params = gymnax.make(args.env_name)
    env = GymnaxContinualEnvWrapper(env)
    if args.env_name.lower() in ['cartpole-v1']:
        env = PenalizeTerminationWrapper(env)
    env = EquinoxEnv(env_key, env, env_params)

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    

    ### Initialize the model ###

    feature_extractor = FeatureExtractor(
        key = fe_key,
        obs_dim = env.observation_space.shape,
        layer_sizes = args.fe_layer_sizes,
        output_dim = args.fe_latent_dim,
        recurrent_layer_indices = args.fe_recurrent_layer_indices,
    )
    model = ActorCriticModel(
        key = ac_key,
        feature_extractor = feature_extractor,
        action_dim = env.action_space.n,
        actor_layer_sizes = args.actor_layer_sizes,
        critic_layer_sizes = args.critic_layer_sizes,
    )


    ### Initialize training state ###

    if args.use_swift_td:
        swift_td_state = SwiftTDState(
            n_features = args.critic_layer_sizes[-1],
            lr_init = args.std_lr_init,
            meta_lr = args.std_meta_lr,
            epsilon = args.std_epsilon,
            eta = args.std_eta,
            trace_decay = args.std_trace_decay,
            gamma = args.std_gamma,
        )
    else:
        swift_td_state = None

    optimizer = adam(args.learning_rate)
    opt_state = optimizer.init(model)

    train_state = TrainState(
        rng = train_key,
        model = model,
        opt_state = opt_state,
        tx_update_fn = optimizer.update,
        obs_shape = env.observation_space.shape,
        gamma = args.gamma,
        feature_update_freq = args.feature_update_freq,
        tbptt_window = args.tbptt_window,
        use_swift_td = args.use_swift_td,
        swift_td_state = swift_td_state,
    )


    ### Training loop setup ###

    env, obs = env.reset()
    rnn_state = model.init_rnn_state()

    train_step_fn = jax.jit(train_loop, static_argnums=(5,))


    # Train
    for train_step in range(args.train_steps // args.log_freq):
        train_state, rnn_state, model, env, obs, metrics = train_step_fn(
            train_state, rnn_state, model, env, obs, args.log_freq)
        print_metrics(train_step * args.log_freq, metrics)


if __name__ == '__main__':
    args = parse_args()
    train(args)
