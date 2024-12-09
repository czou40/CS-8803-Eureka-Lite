def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state
    
    # Reward component for pole being balanced (within 1.57 radians)
    angle_temperature = 10.0
    angle_reward = -np.abs(pole_angle) # Closer to 0 is better
    transformed_angle_reward = np.exp(angle_reward / angle_temperature)
    
    # Reward component for cart moving back and forth between thresholds
    movement_temperature = 0.5
    movement_reward = cart_velocity ** 2 # Faster movements are better
    transformed_movement_reward = np.exp(movement_reward / movement_temperature)
    
    # Check for terminal condition if pole is more than 1.57 radians from vertical
    terminated = np.abs(pole_angle) > 1.57 or cart_position < -4.0 or cart_position > 4.0

    # Truncated is False; not needed as part of regular CartPole problem
    truncated = False

    # Combine rewards
    total_reward = (transformed_angle_reward + transformed_movement_reward)
    
    # Dictionary of each individual reward component
    reward_components = {
        "angle_reward": np.float64(transformed_angle_reward),
        "movement_reward": np.float64(transformed_movement_reward),
    }
    
    return np.float64(total_reward), terminated, truncated, reward_components
