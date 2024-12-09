def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Parameters for reward transformations
    angle_temperature = 1.0
    velocity_temperature = 0.3
    threshold_touch_bonus = 1.0  # Increased for better emphasis

    # Reward for keeping pole balanced, adjusted to focus on within operational range
    angle_reward = np.exp(-angle_temperature * (abs(pole_angle) / 1.57))
    
    # Encourage minimal but sufficient velocity for constructive motion by adjusting temperature
    velocity_reward = np.exp(-velocity_temperature * abs(cart_velocity))
    
    # Improved bonus reward to encourage periodic threshold touching
    threshold_touch_reward = threshold_touch_bonus if has_touched_left_threshold and has_touched_right_threshold else 0.0

    # Termination conditions (episode ends if pole or cart strays beyond limits)
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    # No specific truncation condition mentioned
    truncated = False
    
    # Compute total reward with recalibrated component magnitudes
    total_reward = np.float64(0.7 * angle_reward + 0.2 * velocity_reward + 0.1 * threshold_touch_reward)
    
    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
    }
    
    return total_reward, terminated, truncated, reward_components
