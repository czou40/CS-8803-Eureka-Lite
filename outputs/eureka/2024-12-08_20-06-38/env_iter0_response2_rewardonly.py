def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Reward component for keeping the pole balanced
    angle_constraint = 1.57
    pole_balance_reward = np.cos(pole_angle)  # Reward higher for angles closer to 0 (vertical position)

    # Reward for the cart moving between swing thresholds
    within_swing_range = (cart_position <= swing_threshold) and (cart_position >= -swing_threshold)
    cart_movement_reward = 1.0 if (touched_left or touched_right) and within_swing_range else 0.0

    # Define temperatures for any transformations, if needed
    pole_temp = 1.0
    cart_temp = 1.0

    # Apply transformations, if necessary
    transformed_pole_balance_reward = np.exp(pole_temp * pole_balance_reward)
    transformed_cart_movement_reward = np.exp(cart_temp * cart_movement_reward)

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
