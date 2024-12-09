import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces
import numpy as np

class CustomCartPoleEnv(CartPoleEnv):
    def get_state(self) -> np.ndarray:
        # Compute the observation
        # swing threshold is just a constant that is added to the state for the ease of access.
        return np.array([self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity, self.swing_threshold, 1.0 if self.has_touched_left_threshold else 0.0, 1.0 if self.has_touched_right_threshold else 0.0])
    