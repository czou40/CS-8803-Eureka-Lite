def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Pole angle normalized within [-1.57, 1.57]
    angle_constraint = 1.57
    pole_balance_quality = 1.0 - (abs(pole_angle) / angle_constraint)

    # Encourage staying close to vertical
    pole_balance_reward = 2.0 * pole_balance_quality  

    # Reward for cart moving between thresholds swiftly
    desired_movement = (touched_left and touched_right)
    cart_movement_quality = 1.0 if desired_movement else 0.5
    cart_movement_reward = 1.0 * cart_movement_quality * np.abs(cart_velocity)

    # Temperature parameters for transformation
    pole_temp = 0.3
    cart_temp = 0.3

    # Apply transformations
    transformed_pole_balance_reward = np.exp(pole_temp * pole_balance_reward) - 1
    transformed_cart_movement_reward = np.exp(cart_temp * cart_movement_reward) - 1

    # Total reward - More balanced between components
    reward = np.float64(transformed_pole_balance_reward + transformed_cart_movement_reward)

    # Termination conditions
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > 4.0
    truncated = False

    # Reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'cart_movement_reward': np.float64(transformed_cart_movement_reward)
    }

    return reward, terminated, truncated, reward_components
