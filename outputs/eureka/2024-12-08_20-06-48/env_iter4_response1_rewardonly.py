def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    cart_position = internal_state[0]
    cart_velocity = internal_state[1]
    pole_angle = internal_state[2]
    pole_angular_velocity = internal_state[3]
    swing_threshold = internal_state[4]
    has_touched_left_threshold = internal_state[5] == 1.0
    has_touched_right_threshold = internal_state[6] == 1.0

    # Improved temperature parameters
    angle_temperature = 1.8
    velocity_temperature = 2.2  # more influence to encourage faster swinging
    threshold_touch_bonus = 50.0  # significant bonus to encourage better exploration

    # Angle Reward: Reward for keeping pole close to vertical
    angle_reward = np.exp(-angle_temperature * (abs(pole_angle) / 1.57))

    # Velocity Reward: Adjusted to encourage faster cart movement
    velocity_reward = np.exp(-velocity_temperature * np.tanh(abs(cart_velocity)))

    # Threshold Touch Reward: Encourage cart to reach both ends
    threshold_touch_reward = threshold_touch_bonus if has_touched_left_threshold and has_touched_right_threshold else 0.0

    # Termination conditions
    terminated = abs(pole_angle) > 1.57 or abs(cart_position) > 4.0

    truncated = False

    # Total reward combines weighted components
    angle_weight = 0.5
    velocity_weight = 0.4
    threshold_weight = 0.1
    total_reward = np.float64(angle_weight * angle_reward + 
                              velocity_weight * velocity_reward + 
                              threshold_weight * threshold_touch_reward)

    reward_components = {
        'angle_reward': np.float64(angle_reward),
        'velocity_reward': np.float64(velocity_reward),
        'threshold_touch_reward': np.float64(threshold_touch_reward),
    }

    return total_reward, terminated, truncated, reward_components
