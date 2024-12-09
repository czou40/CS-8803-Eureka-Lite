def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Unpack the state variables
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state

    # Define thresholds for termination
    angle_threshold = 12 * (np.pi / 180)  # 12 degrees
    position_threshold = 2.4

    # Modified pole stability reward using exponential scaling for better differentiation
    angle_temp_factor = 5.0  # New temperature parameter
    pole_stability_reward = np.float64(np.exp(-angle_temp_factor * abs(pole_angle)))

    # Simplify the cart position penalty by moderating with a more modest scale
    cart_position_scale = 1.0  # Scale adjustment
    if abs(cart_position) < position_threshold:
        cart_position_penalty = np.float64(cart_position_scale * (1.0 - abs(cart_position) / position_threshold))
    else:
        cart_position_penalty = np.float64(-10.0)  # Heavy penalty when exceeding threshold

    # Sum the primary rewards with proper scaling
    total_reward = pole_stability_reward + cart_position_penalty

    # Check for termination condition
    terminated = bool(abs(pole_angle) > angle_threshold or abs(cart_position) > position_threshold)
    truncated = False

    # Construct reward breakdown
    reward_components = {
        "pole_stability_reward": pole_stability_reward,
        "cart_position_penalty": cart_position_penalty
    }

    return total_reward, terminated, truncated, reward_components
