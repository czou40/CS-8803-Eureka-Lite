def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Scale parameters
    cart_position_scale = 4.0
    angle_constraint = 1.57

    # Reward for keeping pole upright
    pole_balance_reward = np.exp(-10 * (abs(pole_angle) / angle_constraint)) - 1

    # Reward for cart movement between thresholds
    if touched_left or touched_right:
        cart_movement_reward = 1.0 - np.tanh(0.1 * abs(cart_velocity))
    else:
        cart_movement_reward = 0.0

    # Temperature parameters
    pole_temp = 0.3
    cart_temp = 0.7

    # Transform rewards
    transformed_pole_balance_reward = np.exp(pole_temp * pole_balance_reward) - 1
    transformed_cart_movement_reward = cart_temp * cart_movement_reward 

    # Total reward
    total_reward = np.float64(transformed_pole_balance_reward + transformed_cart_movement_reward)

    # Termination conditions
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > cart_position_scale
    truncated = False

    # Collect individual reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward)
    }

    return total_reward, terminated, truncated, reward_components
