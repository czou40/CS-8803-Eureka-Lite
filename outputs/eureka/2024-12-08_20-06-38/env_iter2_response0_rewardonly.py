def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Scaled reward for keeping the pole as upright as possible
    angle_constraint = 1.57
    normalized_angle = (angle_constraint - abs(pole_angle)) / angle_constraint
    pole_balance_reward = normalized_angle * 0.5  # Scale down by factor of 0.5

    # Reward for cart movement
    if touched_left or touched_right:
        cart_movement_reward = 1.0  # Reward only when reaching thresholds
    else:
        cart_movement_reward = 0.0

    # Incorporate velocity into cart reward to encourage quick movement
    velocity_bonus = min(0.1 * np.abs(cart_velocity), 1.0)
    cart_movement_reward += velocity_bonus

    # Define temperature parameters
    pole_temp = 1.0
    cart_temp = 1.0

    # Apply transformations
    transformed_pole_balance_reward = np.exp(pole_temp * pole_balance_reward) - 1
    transformed_cart_movement_reward = np.exp(cart_temp * cart_movement_reward) - 1

    # Total reward
    reward = np.float64(transformed_pole_balance_reward + transformed_cart_movement_reward)

    # Check termination conditions
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > 4.0
    truncated = False

    # Collect individual reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward)
    }

    return reward, terminated, truncated, reward_components
