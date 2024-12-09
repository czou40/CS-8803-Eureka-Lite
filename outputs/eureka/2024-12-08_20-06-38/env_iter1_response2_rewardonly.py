def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Define constraints
    angle_constraint = 1.57
    position_constraint = 4.0

    # Pole balance reward (scale reduced)
    pole_balance_reward = np.exp(-5 * np.abs(pole_angle))

    # Cart movement reward (adjusted for regular crossings)
    if touched_left or touched_right:
        cart_movement_reward = 1.0
    else:
        cart_movement_reward = 0.1 * (np.abs(cart_position) < swing_threshold)

    # Normalize rewards
    transformed_pole_balance_reward = pole_balance_reward
    transformed_cart_movement_reward = cart_movement_reward

    # Total reward
    reward = np.float64(transformed_pole_balance_reward + transformed_cart_movement_reward)

    # Check termination conditions
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > position_constraint
    truncated = False

    # Collect individual reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward)
    }

    return reward, terminated, truncated, reward_components
