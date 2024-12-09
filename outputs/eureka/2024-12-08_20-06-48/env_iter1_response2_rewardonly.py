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
    velocity_temperature = 0.2
    swing_bonus_temperature = 5.0

    # Reward for keeping pole balanced
    angle_reward = np.exp(-angle_temperature * abs(pole_angle))
    
    # Penalty for moving cart beyond swing threshold
    position_penalty = -1.0 * (abs(cart_position) > swing_threshold)
    
    # Encourage periodic swinging by giving bonus for touching both thresholds frequently
    swing_periodicity_reward = np.exp(-swing_bonus_temperature * abs(cart_position - swing_threshold * (1 if has_touched_right_threshold else -1)))
    
    # Encourage moderate speed but without aggressive movement
    velocity_reward = np.exp(-velocity_temperature * abs(cart_velocity - 0.5 * (cart_velocity > 0) + 0.5 * (cart_velocity < 0)))
    
    total_reward = np.float64(angle_reward + position_penalty + velocity_reward + swing_periodicity_reward)

    # Termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    # No truncation condition is provided
    truncated = False

    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'position_penalty': np.float64(position_penalty),
        'velocity_reward': np.float64(velocity_reward),
        'swing_periodicity_reward': np.float64(swing_periodicity_reward),
    }

    return total_reward, terminated, truncated, reward_components
