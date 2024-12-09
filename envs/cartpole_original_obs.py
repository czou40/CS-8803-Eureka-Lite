import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces
import numpy as np

class CustomCartPoleEnv(CartPoleEnv):
    def get_state(self) -> np.ndarray:
        # Compute the observation
        return np.array([self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity])
