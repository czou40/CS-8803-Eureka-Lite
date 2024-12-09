def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Adjusted temperature parameters and scaling
    angle_temperature = 2.0
    velocity_temperature = 2.0
    touch_temperature = 3.0
    threshold_touch_bonus = 20.0
   
    # Reward for keeping the pole balanced
    angle_reward = np.exp(-angle_temperature * (abs(pole_angle) / 1.57))

    # Reward for cart velocity to encourage quick swings while remaining stable
    velocity_reward = np.exp(-velocity_temperature * min(abs(cart_velocity), 2.0))

    # Bonus for touching thresholds: prioritize bilateral swing threshold touching
    if has_touched_left_threshold and has_touched_right_threshold:
        threshold_touch_reward = threshold_touch_bonus
    else:
        threshold_touch_reward = np.exp(-touch_temperature * abs(min(0.0, cart_position + swing_threshold, cart_position - swing_threshold)))

    # Introduce a component for central position to keep cart balanced
    central_position_reward = np.exp(-1.5 * abs(cart_position) / 4.0)

    # Termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    truncated = False

    # Re-balanced total reward considering importance of components
    total_reward = np.float64(
        0.3 * angle_reward +
        0.3 * velocity_reward +
        0.2 * threshold_touch_reward +
        0.2 * central_position_reward
    )

    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
        'central_position_reward': np.float64(central_position_reward),
    }

    return total_reward, terminated, truncated, reward_components
