def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Re-scaled pole balance reward
    pole_balance_reward = 1.0 - (abs(pole_angle) / 1.57)
    pole_balance_scale = 1.0
    pole_balance_temp = 0.02

    # Adjusted cart movement reward
    cart_transition_reward = 0.2 if (touched_left or touched_right) else -0.1
    cart_movement_scale = 1.0
    cart_movement_temp = 0.05

    # Increased pole velocity penalty
    pole_velocity_penalty = -0.05 * abs(pole_angular_velocity)

    # Scale and transform rewards
    transformed_pole_balance = (np.exp(pole_balance_temp * pole_balance_reward * pole_balance_scale) - 1)
    transformed_cart_movement = (np.exp(cart_movement_temp * cart_transition_reward * cart_movement_scale) - 1)

    # Total reward
    total_reward = np.float64(transformed_pole_balance + transformed_cart_movement + pole_velocity_penalty)

    # Termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0
    truncated = False

    # Reward components dictionary
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance),
        'cart_movement_reward': np.float64(transformed_cart_movement),
        'pole_velocity_penalty': np.float64(pole_velocity_penalty)
    }

    return total_reward, terminated, truncated, reward_components
