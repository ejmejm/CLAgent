from functools import partial
from typing import Optional, Union

import chex
import equinox as eqx
import gymnasium as gym
import gymnax
from gymnax.environments.environment import Environment as gymnaxEnv
from gymnax.wrappers.purerl import GymnaxWrapper
import jax
import jax.numpy as jnp

from .utils import tree_replace


class GymContinualEnvWrapper(gym.Wrapper):
    """Wraps a gym environment to make it continual.
        
    Whenever the `step` function returns `done=True`, the environment is reset
    automatically, and the transition to the intial observation will give no reward.
    """

    def step(self, action: int):
        """Performs a step, and automatically resets when necessary."""
        if self.reset_required:
            obs = self._env.reset()
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
        else:
            obs, reward, terminated, truncated, info = self._env.step(action)

        self.reset_required = terminated

        return obs, reward, terminated, truncated, info 
    
    def reset(self):
        self.reset_required = False
        return self._env.reset()


class GymnaxContinualEnvWrapper(GymnaxWrapper):
    """Wraps a gym environment to make it continual.
    
    Whenever the `step` function returns `done=True`, the environment is reset
    automatically, and the transition to the intial observation will give no reward.
    """

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: gymnax.EnvState,
            action: Union[int, float],
            params: Optional[gymnax.EnvParams] = None,
        ):
        """Performs a step, and automatically resets when necessary."""

        def reset_step(key, params):
            obs, state = self._env.reset(key, params)
            return obs, state, 0.0, False, {}

        # Have options for using and not using JAX because lax.cond is going
        # to be very slow if the environment is not jitted.
        obs, state, reward, done, info = jax.lax.cond(
            self._env.is_terminal(state, params),
            lambda: reset_step(key, params),
            # TODO: Find a way to bring back info
            # Need to make a dummy info for the `reset_step` function
            lambda: self._env.step(key, state, action, params)[:4] + ({},),
        )

        return obs, state, reward, done, info 

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Optional[gymnax.EnvParams] = None):
        obs, state = self._env.reset(key, params)
        return obs, state


class PenalizeTerminationWrapper(GymnaxWrapper):
    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: gymnax.EnvState,
            action: Union[int, float],
            params: Optional[gymnax.EnvParams] = None,
        ):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        reward = jax.lax.cond(
            done,
            lambda: reward - 100.0,
            lambda: reward,
        )
        return obs, state, reward, done, info


class EquinoxEnv(eqx.Module):
    """Wraps a gymnax environment to make it usable without explicit state carrying."""
    rng: chex.PRNGKey
    env: gymnaxEnv = eqx.field(static=True)
    env_params: Optional[gymnax.EnvParams] = eqx.field(static=True)
    env_state: gymnax.EnvState = eqx.field(converter=lambda x: jax.tree.map(jnp.array,x ))

    def __init__(self, rng: chex.PRNGKey, env: gymnaxEnv, env_params: Optional[gymnax.EnvParams] = None):
        reset_key, self.rng = jax.random.split(rng)
        self.env = env
        self.env_params = env_params
        env_state = self.env.reset(reset_key, env_params)
        self.env_state = self._convert_state(env_state)
    
    def reset(self):
        reset_key, rng = jax.random.split(self.rng)
        obs, state = self.env.reset(reset_key, self.env_params)
        new_env = tree_replace(self, env_state=state, rng=rng)
        return new_env, obs

    def step(self, action: int):
        step_key, rng = jax.random.split(self.rng)
        obs, state, reward, done, info = self.env.step(
            step_key, self.env_state, action, self.env_params)
        new_env = tree_replace(self, env_state=state, rng=rng)
        return new_env, obs, reward, done, info
    
    @property
    def observation_space(self):
        return self.env.observation_space(params=self.env_params)

    @property
    def action_space(self):
        return self.env.action_space(params=self.env_params)

    def _convert_state(self, state: gymnax.EnvState):
        return jax.tree.map(jnp.array, state)
    
    def __getattr__(self, name):
        return getattr(self.env, name)