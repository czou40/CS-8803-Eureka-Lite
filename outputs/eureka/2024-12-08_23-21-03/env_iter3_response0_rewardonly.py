def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract relevant variables from the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Define the thresholds for terminating the episode
    angle_threshold = 12 * (np.pi / 180)  # radians
    position_threshold = 2.4  # cart position

    # Check if the episode is terminated 
    terminated = np.abs(pole_angle) > angle_threshold or np.abs(cart_position) > position_threshold
    truncated = False

    # Redefined temperature parameters based on feedback after analysis
    
    # Increase sensitivity and re-scale pole upright reward
    pole_upright_reward_temp = 10.0  # Increased to better optimize keeping the pole upright
    pole_upright_reward = np.exp(-pole_upright_reward_temp * abs(pole_angle))

    # Maintain similar influence for cart centered reward but add slight sensitivity
    cart_centered_reward_temp = 0.5  # Lowered to better optimize around the center
    cart_centered_reward = np.exp(-cart_centered_reward_temp * abs(cart_position))

    # Re-scale pole stability reward to have a higher influence
    pole_stability_reward_temp = 2.0  # Increased to encourage reduced angular velocity
    pole_stability_reward = np.exp(-pole_stability_reward_temp * abs(pole_angular_velocity))

    # Increase sensitivity for cart stability to have a more uniform effect
    cart_stability_reward_temp = 1.0  # Re-scaled for consistent influence
    cart_stability_reward = np.exp(-cart_stability_reward_temp * abs(cart_velocity))

    # Total reward recalibrated based on the new parametrization
    total_reward = (2.5 * pole_upright_reward +
                    2.0 * cart_centered_reward +
                    1.5 * pole_stability_reward +
                    1.5 * cart_stability_reward)

    # Convert reward components to np.float64 for consistency
    reward_components = {
        "pole_upright_reward": np.float64(pole_upright_reward),
        "cart_centered_reward": np.float64(cart_centered_reward),
        "pole_stability_reward": np.float64(pole_stability_reward),
        "cart_stability_reward": np.float64(cart_stability_reward)
    }

    return np.float64(total_reward), terminated, truncated, reward_components
