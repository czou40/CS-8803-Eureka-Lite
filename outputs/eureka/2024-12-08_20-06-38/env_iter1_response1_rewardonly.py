def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Reward for keeping the pole balanced
    angle_constraint = 1.57
    pole_balance_reward = 1.0 - (abs(pole_angle) / angle_constraint)  # Normalize within [0, 1]

    # Reward for cart movement between thresholds
    if touched_left or touched_right:
        cart_movement_reward = 1.0
    else:
        cart_movement_reward = max(0.0, 1.0 - abs(cart_position) / swing_threshold)

    # Penalize proximity to cart boundaries
    boundary_penalty = max(0.0, abs(cart_position) - 3.5)  # Stronger penalty if near boundaries

    # Define temperatures for transformations
    pole_temp = 0.5
    cart_temp = 0.5

    # Apply transformations
    transformed_pole_balance_reward = np.exp(pole_temp * pole_balance_reward)
    transformed_cart_movement_reward = np.exp(cart_temp * cart_movement_reward)

    # Total reward
    reward = np.float64(transformed_pole_balance_reward + transformed_cart_movement_reward - boundary_penalty)

    # Check termination conditions
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > 4.0
    truncated = False

    # Collect individual reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward),
        'boundary_penalty': np.float64(boundary_penalty)
    }

    return reward, terminated, truncated, reward_components
