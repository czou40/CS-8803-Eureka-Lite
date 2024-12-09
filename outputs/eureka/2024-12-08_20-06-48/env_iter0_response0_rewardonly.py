def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Parameters for reward transformations
    angle_temperature = 0.5
    velocity_temperature = 0.1
    threshold_touch_bonus = 0.1

    # Reward for keeping pole balanced (i.e., angle close to zero)
    angle_reward = np.exp(-angle_temperature * abs(pole_angle))
    
    # Reward for cart velocity to encourage movement but not fast enough to lose balance
    velocity_reward = np.exp(-velocity_temperature * abs(cart_velocity))
    
    # Bonus reward for touching both swing thresholds periodicaly and alternating
    threshold_touch_reward = 0.0
    if has_touched_left_threshold and has_touched_right_threshold:
        threshold_touch_reward = threshold_touch_bonus

    # Termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    # There is no specific truncation condition mentioned
    truncated = False
    
    # Total reward
    total_reward = np.float64(angle_reward + velocity_reward + threshold_touch_reward)
    
    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
    }
    
    return total_reward, terminated, truncated, reward_components
