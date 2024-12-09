def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, swing_threshold, touched_left, touched_right = internal_state

    # Balance Reward: Encourage the pole to stay upright near zero angle
    angle_constraint = 1.57
    pole_balance_score = max(0, (angle_constraint - abs(pole_angle)) / angle_constraint)
    pole_balance_reward = pole_balance_score
    
    # Movement Reward: Encourage quick back-and-forth movement
    movement_reward = 0.0
    if touched_left or touched_right:
        movement_reward += 0.5  # Additional incentive for reaching thresholds
    # Encourage higher velocity towards thresholds
    movement_reward += min(0.5 * np.abs(cart_velocity), 1.0)

    # Apply scaling for transformations
    pole_temp = 0.05
    cart_temp = 0.1

    # Transform rewards with scaling adjustments
    transformed_pole_balance_reward = (np.exp(pole_temp * pole_balance_reward) - 1) * 5
    transformed_movement_reward = (np.exp(cart_temp * movement_reward) - 1) * 10

    # Sum all rewards
    total_reward = np.float64(transformed_pole_balance_reward + transformed_movement_reward)

    # Termination logic
    terminated = np.abs(pole_angle) > angle_constraint or np.abs(cart_position) > 4.0
    truncated = False

    # Record individual reward components
    reward_components = {
        'pole_balance_reward': np.float64(transformed_pole_balance_reward),
        'movement_reward': np.float64(transformed_movement_reward)
    }

    return total_reward, terminated, truncated, reward_components
