def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Reward for keeping the pole angle within the limit
    if -1.57 < pole_angle < 1.57:
        balance_reward = np.float64(1.0)
    else:
        balance_reward = np.float64(-1.0)

    # Reward for keeping the cart within the range
    if -4.0 < cart_position < 4.0:
        position_reward = np.float64(0.1)
    else:
        position_reward = np.float64(-1.0)

    # Reward for oscillation between swing_thresholds
    oscillation_reward = np.float64(0.0)
    if touched_left and touched_right:
        oscillation_reward = np.float64(1.0)
    
    # Combined reward calculation
    combined_reward = balance_reward + position_reward + oscillation_reward

    # Apply exponential transformation for normalization
    balance_temperature = 0.5
    position_temperature = 0.2
    oscillation_temperature = 0.3
    
    balance_reward_transformed = torch.exp(balance_temperature * balance_reward)
    position_reward_transformed = torch.exp(position_temperature * position_reward)
    oscillation_reward_transformed = torch.exp(oscillation_temperature * oscillation_reward)

    total_reward = np.float64(balance_reward_transformed + position_reward_transformed + oscillation_reward_transformed)

    # Check termination conditions
    terminated = not (-1.57 < pole_angle < 1.57 or -4.0 < cart_position < 4.0)
    truncated = False  # Not truncating episodes based on time for this task

    # Reward components dictionary
    reward_components = {
        "balance_reward": balance_reward,
        "position_reward": position_reward,
        "oscillation_reward": oscillation_reward,
    }

    return total_reward, terminated, truncated, reward_components
