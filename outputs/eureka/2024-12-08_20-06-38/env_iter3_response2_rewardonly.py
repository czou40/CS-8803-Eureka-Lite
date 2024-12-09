def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Pole balance reward scaled and smoothed
    angle_constraint = 1.57
    pole_balance_reward = (angle_constraint - abs(pole_angle)) / angle_constraint
    pole_balance_temp = 2.0
    transformed_pole_balance_reward = np.exp(pole_balance_temp * pole_balance_reward) - 1

    # Cart movement reward when crossing thresholds efficiently
    cart_movement_reward = 0
    cart_movement_temp = 0.5
    if touched_left or touched_right:
        cart_movement_reward += 0.5
        
    # Adding a bonus for high velocity to encourage quick transitions
    velocity_bonus = 0.1 * min(np.abs(cart_velocity), 2.0)
    cart_movement_reward += velocity_bonus

    transformed_cart_movement_reward = np.exp(cart_movement_temp * cart_movement_reward) - 1

    # Total reward
    reward = np.float64(transformed_pole_balance_reward + 2 * transformed_cart_movement_reward)

    # Check for termination
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > 4.0
    truncated = False

    # Individual reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward)
    }

    return reward, terminated, truncated, reward_components
