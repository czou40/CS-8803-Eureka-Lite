def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract relevant variables from the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Define the thresholds for terminating the episode
    angle_threshold = 12 * (np.pi / 180)  # radians
    position_threshold = 2.4  # cart position

    # Check if the episode is terminated 
    terminated = np.abs(pole_angle) > angle_threshold or np.abs(cart_position) > position_threshold
    truncated = False

    # Revise temperature parameters for scaling different components better
    
    # Re-scale pole upright reward to improve learning convergence
    pole_upright_reward_temp = 5.0  # Adjusted for better balance around upright position
    pole_upright_reward = np.exp(-pole_upright_reward_temp * abs(pole_angle))

    # Re-scale cart-centered reward to add more influence when far from center
    cart_centered_reward_temp = 0.1  # Adjusted to reflect position stability's importance
    cart_centered_reward = np.exp(-cart_centered_reward_temp * abs(cart_position))

    # Increase sensitivity for pole stability to enhance angular velocity control
    pole_stability_reward_temp = 3.0  # Increased sensitivity for stability
    pole_stability_reward = np.exp(-pole_stability_reward_temp * abs(pole_angular_velocity))

    # Lower influence of cart velocity to balance other components
    cart_stability_reward_temp = 0.5  # Lowered sensitivity for maintained trajectory
    cart_stability_reward = np.exp(-cart_stability_reward_temp * abs(cart_velocity))

    # Total reward balanced by recalibration of component weights
    total_reward = (3.0 * pole_upright_reward +
                    2.5 * cart_centered_reward +
                    2.0 * pole_stability_reward +
                    1.0 * cart_stability_reward)

    # Convert reward components to np.float64 for consistency
    reward_components = {
        "pole_upright_reward": np.float64(pole_upright_reward),
        "cart_centered_reward": np.float64(cart_centered_reward),
        "pole_stability_reward": np.float64(pole_stability_reward),
        "cart_stability_reward": np.float64(cart_stability_reward)
    }

    return np.float64(total_reward), terminated, truncated, reward_components
