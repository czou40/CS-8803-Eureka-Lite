def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Pole balance reward: Encourage small pole angles
    angle_constraint = 1.57
    pole_balance_score = max(0, (angle_constraint - abs(pole_angle)) / angle_constraint)
    pole_balance_reward = pole_balance_score * 1.0

    # Incremental movement reward: Focus on speed between thresholds
    cart_movement_reward = 0.0
    if touched_left or touched_right:
        cart_movement_reward += 1.0
    cart_movement_reward += min(0.5 * np.abs(cart_velocity), 1.0)

    # Define temperature parameters for transformations
    pole_temp = 0.5
    cart_temp = 0.5

    # Apply transformations
    transformed_pole_balance_reward = np.exp(pole_temp * pole_balance_reward) - 1
    transformed_cart_movement_reward = np.exp(cart_temp * cart_movement_reward) - 1

    # Total reward
    total_reward = np.float64(transformed_pole_balance_reward + transformed_cart_movement_reward)

    # Check termination conditions
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > 4.0
    truncated = False

    # Collect individual reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward)
    }

    return total_reward, terminated, truncated, reward_components
