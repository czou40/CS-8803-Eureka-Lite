def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Updated temperature parameters and scaling
    angle_temperature = 3.0
    velocity_temperature = 1.0
    touch_temperature = 0.05

    # Adjusted rewards
    # Reward for keeping the pole balanced
    angle_reward = np.exp(-angle_temperature * (abs(pole_angle) / 1.57))

    # Encourage higher velocity by reducing damping
    velocity_reward = np.exp(-velocity_temperature * abs(cart_velocity / 4.0))

    # Reward for staying within the swing thresholds using a smoother exponential decay
    proximity_to_threshold = max(0.0, 1.0 - abs(cart_position / swing_threshold))
    threshold_touch_reward = np.exp(-touch_temperature * (1 - proximity_to_threshold))

    # Maintaining termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0
    truncated = False

    # Rescale total reward to optimize performance
    total_reward = np.float64(0.4 * angle_reward + 0.3 * velocity_reward + 0.3 * threshold_touch_reward)

    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
    }

    return total_reward, terminated, truncated, reward_components
