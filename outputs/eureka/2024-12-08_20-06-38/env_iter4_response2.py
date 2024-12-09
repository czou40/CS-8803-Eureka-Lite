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
    def __init__(self, swing_threshold = 0.5, max_time_steps=2000, **kwargs):
        super().__init__(**kwargs)

        # Override or store new parameters
        self.swing_threshold = swing_threshold
        self.max_time_steps = max_time_steps
        self.observation_space = spaces.Box(
            low=np.array([-4.8, -np.finfo(np.float64).max, -2.0, -np.finfo(np.float64).max, self.swing_threshold, 0.0, 0.0], dtype=np.float64),
            high=np.array([4.8, np.finfo(np.float64).max, 2.0, np.finfo(np.float64).max, self.swing_threshold, 1.0, 1.0], dtype=np.float64),
            dtype=np.float64
        )
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
        state, self.info = super().reset()
        
        self.reward = np.float64(0.0)
        self.terminated = False
        self.truncated = False
        self.rew_dict: dict[str, np.float64] = {}
        self.rew_dict_cumulative: dict[str, np.float64] = {}
        self.time_step = np.float64(0.0)
        self.gt_reward = np.float64(0.0)
        self.has_touched_left_threshold = False
        self.has_touched_right_threshold = False
        self.has_been_rewarded_for_touching_left_threshold = False
        self.has_been_rewarded_for_touching_right_threshold = False
        self.internal_state = np.array([state[0], state[1], state[2], state[3], self.swing_threshold, 0.0, 0.0])
        return self.internal_state, self.info
    
    def get_state(self) -> np.ndarray:
        # Compute the observation
        # swing threshold is just a constant that is added to the state for the ease of access.
        return np.array([self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity, self.swing_threshold, 1.0 if self.has_touched_left_threshold else 0.0, 1.0 if self.has_touched_right_threshold else 0.0])
    
    def compute_reward(self):
        self.reward, self.terminated, self.truncated, self.rew_dict = compute_reward(self.internal_state)
        pass

    def get_log_summary(self):
        return self.rew_dict_cumulative

    def step(self, action): # type: ignore
        state, _, _, _, self.info = super().step(action)

        # the following code is manually added to update the internal state
        if self.cart_position < -self.swing_threshold:
            self.has_touched_left_threshold = True
        if self.cart_position > self.swing_threshold:
            self.has_touched_right_threshold = True
        self.internal_state = np.array([state[0], state[1], state[2], state[3], self.swing_threshold, 1.0 if self.has_touched_left_threshold else 0.0, 1.0 if self.has_touched_right_threshold else 0.0])

        self.compute_reward()
        self.time_step += 1


        # the following code is manually added as "ground-truth reward" (sparse signal akin to the fitness function)
        if self.has_touched_left_threshold and self.has_touched_right_threshold:
            self.gt_reward = 100.0
            self.has_touched_left_threshold = False
            self.has_touched_right_threshold = False
            self.has_been_rewarded_for_touching_left_threshold = False
            self.has_been_rewarded_for_touching_right_threshold = False
        elif self.has_touched_left_threshold and not self.has_been_rewarded_for_touching_left_threshold:
            self.gt_reward = 10.0
            self.has_been_rewarded_for_touching_left_threshold = True
        elif self.has_touched_right_threshold and not self.has_been_rewarded_for_touching_right_threshold:
            self.gt_reward = 10.0
            self.has_been_rewarded_for_touching_right_threshold = True
        else:
            self.gt_reward = 1.0

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
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Pole balance reward: Encourage small pole angles
    angle_constraint = 1.57
    pole_balance_score = max(0, (angle_constraint - abs(pole_angle)) / angle_constraint)
    pole_balance_reward = 2.0 * pole_balance_score

    # Cart movement reward: Reward for reaching threshold and speed
    cart_movement_reward = 0.0
    if touched_left or touched_right:
        cart_movement_reward += 0.5  # Reduced weight for threshold crossing
    cart_movement_reward += min(0.3 * np.abs(cart_velocity), 1.0)

    # Redefined temperature parameters for transformations
    pole_temp = 0.1
    cart_temp = 0.1

    # Apply transformations
    transformed_pole_balance_reward = (np.exp(pole_temp * pole_balance_reward) - 1) * 10
    transformed_cart_movement_reward = (np.exp(cart_temp * cart_movement_reward) - 1) * 10

    # Total reward
    total_reward = np.float64(transformed_pole_balance_reward + transformed_cart_movement_reward)

    # Check termination conditions
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > 4.0
    truncated = False

    # Collect individual reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward)
    }

    return total_reward, terminated, truncated, reward_components
