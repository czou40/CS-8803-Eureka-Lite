def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract relevant variables from the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Termination thresholds
    angle_threshold = 12 * (np.pi / 180)
    position_threshold = 2.4

    # Determine if the episode should terminate
    terminated = np.abs(pole_angle) > angle_threshold or np.abs(cart_position) > position_threshold
    truncated = False
    
    # Updated temperature parameters to improve learning and stability
    # If necessary, scale individual reward components to normalize their effects
    pole_upright_temp = 10.0
    cart_centered_temp = 1.5
    pole_stability_temp = 1.0
    cart_stability_temp = 1.0

    # Reward component for keeping the pole upright: increased sensitivity by modifying temperature
    pole_upright_reward = np.exp(-pole_upright_temp * np.abs(pole_angle))
    
    # Adjusted reward component for keeping cart centered on the track
    cart_centered_reward = np.exp(-cart_centered_temp * np.abs(cart_position))

    # Reward component for maintaining low angular velocity for stability: re-scaled
    pole_stability_reward = np.exp(-pole_stability_temp * np.abs(pole_angular_velocity))

    # Reward component for maintaining low cart velocity: re-scaled
    cart_stability_reward = np.exp(-cart_stability_temp * np.abs(cart_velocity))

    # Calculate the total reward using updated weights based on feedback
    total_reward = (pole_upright_reward * 5.0 +
                    cart_centered_reward * 2.0 +
                    pole_stability_reward * 1.0 +
                    cart_stability_reward * 1.0)

    # Create a dictionary of individual reward components
    reward_components = {
        "pole_upright_reward": np.float64(pole_upright_reward),
        "cart_centered_reward": np.float64(cart_centered_reward),
        "pole_stability_reward": np.float64(pole_stability_reward),
        "cart_stability_reward": np.float64(cart_stability_reward)
    }

    return np.float64(total_reward), terminated, truncated, reward_components
