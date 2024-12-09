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
    # Extract the specific state variables for ease of use
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Define thresholds and weights for each component
    angle_threshold = np.radians(15)  # Angle threshold in radians (equivalent to +-15 degrees)
    position_threshold = 2.4  # Cart position threshold
    
    # Reward component for maintaining a low pole angle
    angle_temp = 5.0
    if abs(pole_angle) <= angle_threshold:
        angle_reward = np.exp(-angle_temp * (pole_angle / angle_threshold)**2)
    else:
        angle_reward = 0.0

    # Punish for the cart moving too far from the center
    position_temp = 5.0
    if abs(cart_position) <= position_threshold:
        position_reward = np.exp(-position_temp * (cart_position / position_threshold)**2)
    else:
        position_reward = 0.0

    # Encourage low velocities
    velocity_temp = 0.1
    velocity_reward = np.exp(-velocity_temp * (cart_velocity**2 + pole_angular_velocity**2))

    # Total reward is some weighted combination of these components
    total_reward = (angle_reward + position_reward + velocity_reward) / 3.0
    
    # Episode terminates if the pole angle or cart position goes out of the defined boundaries
    terminated = abs(pole_angle) > angle_threshold or abs(cart_position) > position_threshold
    truncated = False  # Truncation logic depends on other factors like maximum step count, not state
    
    # Collect individual reward components
    reward_components = {
        "angle_reward": np.float64(angle_reward),
        "position_reward": np.float64(position_reward),
        "velocity_reward": np.float64(velocity_reward)
    }
    
    return np.float64(total_reward), terminated, truncated, reward_components
