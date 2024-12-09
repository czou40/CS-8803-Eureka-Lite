import math
import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces
import numpy as np
from torch import cartesian_prod

class CustomCartPoleEnv(CartPoleEnv):
    """
    A custom CartPole environment.
    """
    def __init__(self, swing_threshold = 1.0, max_time_steps=500, **kwargs):
        super().__init__(**kwargs)
        # Override or store new parameters
        self.swing_threshold = swing_threshold
        self.max_time_steps = max_time_steps
        self.reset()
        
    @property
    def cart_position(self):
        assert self.internal_state is not None
        return self.internal_state[0]
    
    @property
    def cart_velocity(self):
        assert self.internal_state is not None
        return self.internal_state[1]
    
    @property
    def pole_angle(self):
        assert self.internal_state is not None
        return self.internal_state[2]
    
    @property
    def pole_angular_velocity(self):
        assert self.internal_state is not None
        return self.internal_state[3]
    
    def reset(self, *, seed=None, options=None):
        # Use parent class's reset
        self.internal_state, self.info = super().reset()
        self.reward = np.float64(0.0)
        self.terminated = False
        self.truncated = False
        self.rew_dict: dict[str, np.float64] = {}
        self.rew_dict_cumulative: dict[str, np.float64] = {}
        self.time_step = np.float64(0.0)
        self.gt_reward = np.float64(0.0)
        self.has_touched_left_threshold = False
        self.has_touched_right_threshold = False
        return self.internal_state, self.info
    
    def get_state(self) -> np.ndarray:
        # Compute the observation
        return np.array([self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity])
    
    def compute_reward(self):
        self.reward, self.terminated, self.truncated, self.rew_dict = compute_reward(self.internal_state)
        pass

    def get_log_summary(self):
        return self.rew_dict_cumulative

    def step(self, action): # type: ignore
        self.internal_state, self.gt_reward, _, _, self.info = super().step(action)
        self.compute_reward()
        self.time_step += 1

        # the following code is manually added to update the cumulative reward dictionary
        prev_cumulative_gpt_reward = self.rew_dict_cumulative.get("gpt_reward", np.float64(0.0))
        prev_cumulative_gt_reward = self.rew_dict_cumulative.get("gt_reward", np.float64(0.0))
        self.rew_dict_cumulative = {k: self.rew_dict_cumulative.get(k, np.float64(0.0)) + v for k, v in self.rew_dict.items()}
        self.rew_dict_cumulative["gpt_reward"] = prev_cumulative_gpt_reward + self.reward
        self.rew_dict_cumulative["gt_reward"] = prev_cumulative_gt_reward + self.gt_reward
        self.rew_dict_cumulative["episode_length"] = self.time_step

        return self.internal_state, float(self.reward), self.terminated, self.truncated or self.time_step >= self.max_time_steps, self.info 

from typing import Tuple, Dict
import math
import numpy as np
from numpy import ndarray, float64
def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract relevant variables from the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Define the thresholds for terminating the episode
    angle_threshold = 12 * (np.pi / 180)  # radians
    position_threshold = 2.4  # cart position

    # Check if the episode is terminated 
    terminated = np.abs(pole_angle) > angle_threshold or np.abs(cart_position) > position_threshold
    truncated = False

    # Redefined temperature parameters based on feedback after analysis
    
    # Increase sensitivity and re-scale pole upright reward
    pole_upright_reward_temp = 10.0  # Increased to better optimize keeping the pole upright
    pole_upright_reward = np.exp(-pole_upright_reward_temp * abs(pole_angle))

    # Maintain similar influence for cart centered reward but add slight sensitivity
    cart_centered_reward_temp = 0.5  # Lowered to better optimize around the center
    cart_centered_reward = np.exp(-cart_centered_reward_temp * abs(cart_position))

    # Re-scale pole stability reward to have a higher influence
    pole_stability_reward_temp = 2.0  # Increased to encourage reduced angular velocity
    pole_stability_reward = np.exp(-pole_stability_reward_temp * abs(pole_angular_velocity))

    # Increase sensitivity for cart stability to have a more uniform effect
    cart_stability_reward_temp = 1.0  # Re-scaled for consistent influence
    cart_stability_reward = np.exp(-cart_stability_reward_temp * abs(cart_velocity))

    # Total reward recalibrated based on the new parametrization
    total_reward = (2.5 * pole_upright_reward +
                    2.0 * cart_centered_reward +
                    1.5 * pole_stability_reward +
                    1.5 * cart_stability_reward)

    # Convert reward components to np.float64 for consistency
    reward_components = {
        "pole_upright_reward": np.float64(pole_upright_reward),
        "cart_centered_reward": np.float64(cart_centered_reward),
        "pole_stability_reward": np.float64(pole_stability_reward),
        "cart_stability_reward": np.float64(cart_stability_reward)
    }

    return np.float64(total_reward), terminated, truncated, reward_components
