def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract components from internal_state
    cart_position = internal_state[0]
    pole_angle = internal_state[2]
    swing_threshold = internal_state[4]
    left_threshold_touched = internal_state[5]
    right_threshold_touched = internal_state[6]

    # Initialize reward components
    balance_reward = 0.0
    swing_reward = 0.0
    fail_penalty = 0.0
    
    # Parameters
    temperature_balance = 0.1
    temperature_swing = 0.1
    
    # Reward for keeping the pole within 1.57 radians of the vertical
    if np.abs(pole_angle) <= 1.57:
        balance_reward = np.exp(-temperature_balance * np.abs(pole_angle))
    
    # Reward for touching both thresholds periodically
    if left_threshold_touched and right_threshold_touched:
        swing_reward = np.exp(-temperature_swing * np.abs(cart_position - swing_threshold))
    
    # Penalty for failing the task
    if np.abs(pole_angle) > 1.57 or np.abs(cart_position) > 4.0:
        fail_penalty = -1.0
    
    # Calculate total reward
    total_reward = balance_reward + swing_reward + fail_penalty
    
    # Determine if the episode is terminated or truncated
    terminated = np.abs(pole_angle) > 1.57 or np.abs(cart_position) > 4.0
    truncated = False  # Task description does not mention conditions for truncation
    
    # Create dictionary of each individual reward component
    reward_components = {
        "balance_reward": np.float64(balance_reward),
        "swing_reward": np.float64(swing_reward),
        "fail_penalty": np.float64(fail_penalty)
    }
    
    return np.float64(total_reward), terminated, truncated, reward_components
