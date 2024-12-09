def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Unpack the internal state variables
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left_threshold, touched_right_threshold = internal_state

    # Check if the episode is terminated
    terminated = np.abs(pole_angle) > 1.57 or np.abs(cart_position) > 4.0

    # Initialize components of the reward
    pole_balance_reward = 0.0
    periodic_movement_reward = 0.0

    # Define reward components
    # 1. Reward for keeping the pole balanced
    if np.abs(pole_angle) <= 1.57:
        pole_balance_reward = (1.57 - np.abs(pole_angle)) / 1.57  # Normalized reward component (0 to 1)

    # 2. Reward for periodic movement
    periodic_movement_temperature = 0.5  # Temperature parameter for transformation
    if touched_left_threshold and touched_right_threshold:
        periodic_movement_reward = torch.exp(-periodic_movement_temperature * np.abs(cart_velocity))

    # Total reward is the weighted sum of individual components
    total_reward = pole_balance_reward + periodic_movement_reward

    # Convert the components and total reward to np.float64
    pole_balance_reward = np.float64(pole_balance_reward)
    periodic_movement_reward = np.float64(periodic_movement_reward)
    total_reward = np.float64(total_reward)

    # Whether the episode is truncated (not applicable here, considered False)
    truncated = False

    # Create the reward components dictionary
    reward_components = {
        "pole_balance_reward": pole_balance_reward,
        "periodic_movement_reward": periodic_movement_reward
    }

    return total_reward, terminated, truncated, reward_components
