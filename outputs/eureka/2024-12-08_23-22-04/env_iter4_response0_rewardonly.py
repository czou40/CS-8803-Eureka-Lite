def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Unpack the state variables
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state

    # Define thresholds for termination
    angle_threshold = 12 * (np.pi / 180)  # 12 degrees in radians
    position_threshold = 2.4

    # Adjust pole stability reward, making it more sensitive
    pole_stability_temperature = 1.5  # New temperature for transformation
    pole_stability_reward = np.float64(1.0 - (abs(pole_angle) / angle_threshold))
    pole_stability_reward_transformed = np.float64(np.exp(-pole_stability_temperature * abs(pole_angle)))

    # Increase cart position penalty contribution
    cart_position_scale = 2.0  # New, increased scale factor
    cart_position_penalty = np.float64(-cart_position_scale * abs(cart_position / position_threshold))

    # Normalize the cart position penalty to ensure reward is within a balanced range
    cart_position_penalty = max(min(cart_position_penalty, 0), -1)

    # Combination that emphasizes balance
    stability_importance = 0.7
    position_importance = 0.3

    # Calculate total reward
    total_reward = (
        stability_importance * pole_stability_reward_transformed +
        position_importance * cart_position_penalty
    )

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
