def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract relevant variables from the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Define thresholds for episode termination
    angle_threshold = 12 * (np.pi / 180)  # radians
    position_threshold = 2.4  # cart position
    
    # Check if the episode is terminated 
    terminated = np.abs(pole_angle) > angle_threshold or np.abs(cart_position) > position_threshold
    truncated = False
    
    # Define new temperature parameters for scaling
    pole_upright_temp = 12.0
    cart_centered_temp = 1.0
    pole_stability_temp = 1.5
    cart_stability_temp = 0.7
    
    # Calculate reward components
    pole_upright_reward = np.exp(-pole_upright_temp * abs(pole_angle))
    cart_centered_reward = np.exp(-cart_centered_temp * abs(cart_position))
    pole_stability_reward = np.exp(-pole_stability_temp * abs(pole_angular_velocity))
    cart_stability_reward = np.exp(-cart_stability_temp * abs(cart_velocity))
    
    # Total reward recalibrated based on the new parametrization
    total_reward = (1.5 * pole_upright_reward +
                    1.0 * cart_centered_reward +
                    1.0 * pole_stability_reward +
                    0.5 * cart_stability_reward)
    
    # Convert reward components to np.float64 for consistency
    reward_components = {
        "pole_upright_reward": np.float64(pole_upright_reward),
        "cart_centered_reward": np.float64(cart_centered_reward),
        "pole_stability_reward": np.float64(pole_stability_reward),
        "cart_stability_reward": np.float64(cart_stability_reward)
    }

    return np.float64(total_reward), terminated, truncated, reward_components
