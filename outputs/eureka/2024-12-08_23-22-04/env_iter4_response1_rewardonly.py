def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Unpack the state variables
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state

    # Define thresholds for termination
    angle_threshold = 12 * (np.pi / 180)  # 12 degrees in radians
    position_threshold = 2.4

    # Adjust pole stability reward for better sensitivity
    pole_stability_temperature = 1.5  # Adjusted for sensitivity
    pole_stability_reward = np.float64(1.0 - (abs(pole_angle) / angle_threshold))
    pole_stability_reward_transformed = np.float64(np.exp(pole_stability_temperature * pole_stability_reward) - 1)

    # Cart position penalty with adapted scaling
    cart_position_scale = 2.0  # Adjusted scale factor
    cart_position_penalty = np.float64(-cart_position_scale * abs(cart_position / position_threshold))

    # Calculate total reward with more emphasis on balanced scaling
    total_reward = pole_stability_reward_transformed + cart_position_penalty

    # Check for termination conditions
    terminated = bool(abs(pole_angle) > angle_threshold or abs(cart_position) > position_threshold)
    truncated = False  # Assume no episode truncation

    # Compile reward components
    reward_components = {
        "pole_stability_reward": pole_stability_reward,
        "pole_stability_reward_transformed": pole_stability_reward_transformed,
        "cart_position_penalty": cart_position_penalty
    }

    return total_reward, terminated, truncated, reward_components
