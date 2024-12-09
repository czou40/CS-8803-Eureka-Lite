def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Unpack the state variables
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state

    # Define thresholds for termination
    angle_threshold = 12 * (np.pi / 180)  # 12 degrees in radians
    position_threshold = 2.4

    # Initialize reward components
    pole_stability_reward = np.float64(1.0 - (abs(pole_angle) / angle_threshold))
    
    # Apply transformation with adjusted temperature
    pole_stability_temperature = 0.5  # Lower temperature for increased sensitivity
    pole_stability_reward_transformed = np.float64(np.exp(pole_stability_reward / pole_stability_temperature))
    
    # Rescale cart position penalty
    cart_position_scale = 0.1  # Scale factor to reduce its impact
    cart_position_penalty = np.float64(-cart_position_scale * abs(cart_position))

    # Combine reward components
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
