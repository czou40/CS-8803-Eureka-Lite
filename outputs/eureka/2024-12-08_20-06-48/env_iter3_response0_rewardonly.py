def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Updated temperature parameters for better scaling
    angle_temperature = 1.0
    velocity_temperature = 0.5
    threshold_touch_bonus = 5.0

    # Adjusted reward for keeping the pole balanced, reward should encourage near zero angle
    angle_reward = np.exp(-angle_temperature * (abs(pole_angle)))

    # Reward for encouraging high velocity swings for touching thresholds
    cart_velocity_scale = 0.1
    velocity_reward = np.exp(-velocity_temperature * abs(cart_velocity * cart_velocity_scale))

    # Substantial bonus for touching both thresholds during swings
    threshold_touch_reward = threshold_touch_bonus if has_touched_left_threshold and has_touched_right_threshold else 0.0

    # Termination conditions if pole's angle exceeds limits or cart position is out of bounds
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    truncated = False

    # Total reward combines weighted components with adjusted weights
    total_reward = np.float64(0.4 * angle_reward + 0.3 * velocity_reward + 0.3 * threshold_touch_reward)

    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
    }

    return total_reward, terminated, truncated, reward_components
