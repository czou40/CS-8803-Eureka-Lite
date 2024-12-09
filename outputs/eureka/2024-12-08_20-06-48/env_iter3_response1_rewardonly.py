def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Updated temperature parameters
    angle_temperature = 2.0
    velocity_temperature = 1.5  # lower temp for more influence
    threshold_touch_bonus = 10.0  # increased bonus

    # Reward for keeping the pole balanced
    angle_reward = np.exp(-angle_temperature * (abs(pole_angle) / 1.57))

    # Reward for encouraging high velocity swings
    cart_velocity_scale = 0.1  # increase the scale for better swing visualization
    velocity_reward = np.exp(-velocity_temperature * cart_velocity_scale * abs(cart_velocity))

    # Bonus for touching both thresholds during swing
    threshold_touch_reward = threshold_touch_bonus if has_touched_left_threshold and has_touched_right_threshold else 0.0

    # Termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    truncated = False

    # Total reward combines weighted components
    total_reward = np.float64(0.6 * angle_reward + 0.3 * velocity_reward + 0.1 * threshold_touch_reward)

    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
    }

    return total_reward, terminated, truncated, reward_components