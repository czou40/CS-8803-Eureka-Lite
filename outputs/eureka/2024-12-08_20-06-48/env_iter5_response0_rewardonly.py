def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Updated temperature parameters for better sensitivity
    angle_temperature = 2.5  # slightly reduced to sharpen angle sensitivity
    velocity_temperature = 1.8  # reduced for effective velocity optimization
    threshold_touch_weight = 25.0  # Increased influence significantly

    # Reward for keeping the pole balanced
    angle_reward = np.exp(-angle_temperature * (abs(pole_angle) / 1.57))

    # Reward for encouraging high velocity swings
    # Revised velocity scaling for further impact
    velocity_scaling = 0.2 
    velocity_reward = np.exp(-velocity_temperature * velocity_scaling * np.clip(abs(cart_velocity), 0, 10))

    # Drive to push touching both boundaries, now with noticeable increased award
    if has_touched_left_threshold and has_touched_right_threshold:
        threshold_touch_reward = threshold_touch_weight
    else:
        threshold_touch_reward = 0.0

    # Termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    truncated = False

    # Total reward with balanced components
    total_reward = np.float64(0.4 * angle_reward + 0.3 * velocity_reward + 0.3 * threshold_touch_reward)

    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
    }

    return total_reward, terminated, truncated, reward_components
