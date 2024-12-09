def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Reward for keeping the pole as upright as possible
    angle_constraint = 1.57
    normalized_angle = (angle_constraint - abs(pole_angle)) / angle_constraint  # Normalized to [0, 1]
    pole_balance_reward = normalized_angle

    # Reward for cart movement
    within_swing_range = (cart_position <= swing_threshold) and (cart_position >= -swing_threshold)
    cart_movement_reward = np.abs(cart_velocity) if within_swing_range else 0.0

    # Define temperature parameters
    pole_temp = 0.5
    cart_temp = 0.1

    # Apply transformations
    transformed_pole_balance_reward = np.exp(pole_temp * pole_balance_reward) - 1  # Offset to ensure non-zero starting point
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
