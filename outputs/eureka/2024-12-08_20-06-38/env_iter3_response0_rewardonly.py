def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Reward for pole balance
    angle_constraint = 1.57
    normalized_angle = (angle_constraint - abs(pole_angle)) / angle_constraint
    pole_balance_reward = normalized_angle * 0.2  # Reduced scaling for balance reward

    # Reward for periodic cart movement, adjusted for peak and velocity
    movement_peak_reward = 1.0 if touched_left or touched_right else 0.0
    cart_velocity_contribution = min(0.05 * np.abs(cart_velocity), 0.5)
    cart_movement_reward = movement_peak_reward + cart_velocity_contribution

    # Temperature scaling parameters
    pole_temp = 0.5
    cart_temp = 0.2

    # Apply transformations
    transformed_pole_balance_reward = np.exp(pole_temp * pole_balance_reward) - 1
    transformed_cart_movement_reward = np.exp(cart_temp * cart_movement_reward) - 1

    # Total reward calculation
    reward = np.float64(transformed_pole_balance_reward + transformed_cart_movement_reward)

    # Check termination conditions
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > 4.0
    truncated = False

    # Reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward)
    }

    return reward, terminated, truncated, reward_components
