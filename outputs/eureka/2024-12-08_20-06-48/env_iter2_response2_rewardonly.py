def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Adjusted temperature parameters to highlight under-performing components
    angle_temperature = 0.5  # Penalize angle deviations more heavily as it directly relates to task success
    velocity_temperature = 1.0  # Improve clarity in velocity reward transformation
    threshold_touch_bonus = 1.0  # Encourage touching both thresholds

    # Angle reward: Encourage staying upright but re-scaled since it's overpowered
    angle_reward = np.exp(-angle_temperature * abs(pole_angle))

    # Velocity reward: Optimize cart movement without oscillating too quickly
    velocity_reward = np.exp(-velocity_temperature * (abs(cart_velocity - 1.0)))

    # Threshold touch: Provide a bonus for touching both thresholds
    threshold_touch_reward = 0.0
    if has_touched_left_threshold and has_touched_right_threshold:
        threshold_touch_reward = threshold_touch_bonus

    # Additional reward for keeping angular velocity low;
    # steep penalties for rapid movements that can destabilize
    angular_velocity_temperature = 0.3
    angular_velocity_reward = np.exp(-angular_velocity_temperature * abs(pole_angular_velocity))

    # Termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    truncated = False

    # Total reward combines weighted components, re-normalized to prevent reward inflation
    total_reward = np.float64(angle_reward + velocity_reward + threshold_touch_reward + angular_velocity_reward)

    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
        'angular_velocity_reward': np.float64(angular_velocity_reward),
    }

    return total_reward, terminated, truncated, reward_components
