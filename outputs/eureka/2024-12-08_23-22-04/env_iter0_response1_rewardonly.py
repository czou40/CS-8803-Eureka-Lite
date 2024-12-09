def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Unpack the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state

    # Define termination conditions
    terminated = np.abs(pole_angle) > np.pi / 6 or np.abs(cart_position) > 2.4
    
    # Truncation handling (we assume no truncation in this environment)
    truncated = False

    # Reward components
    max_pole_angle = np.pi / 6
    max_cart_position = 2.4
    angle_stability_reward = np.exp(-0.5 * (pole_angle / max_pole_angle) ** 2)
    angle_stability_temperature = 0.1
    
    position_stability_reward = np.exp(-0.5 * (cart_position / max_cart_position) ** 2)
    position_stability_temperature = 0.1

    # Total reward
    total_reward = np.float64(angle_stability_temperature * angle_stability_reward +
                              position_stability_temperature * position_stability_reward)

    # Reward components dictionary
    reward_components = {
        "angle_stability_reward": np.float64(angle_stability_reward),
        "position_stability_reward": np.float64(position_stability_reward)
    }

    return total_reward, terminated, truncated, reward_components
