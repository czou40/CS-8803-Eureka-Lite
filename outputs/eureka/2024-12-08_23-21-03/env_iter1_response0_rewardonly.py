def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract relevant variables from the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Define the thresholds for terminating the episode
    angle_threshold = 12 * (np.pi / 180)  # radians
    position_threshold = 2.4  # cart position

    # Check if the episode is terminated 
    terminated = np.abs(pole_angle) > angle_threshold or np.abs(cart_position) > position_threshold
    truncated = False

    # Re-evaluating and adjusting the reward components
    # Normalize to encourage balancing within limits and slight adjustments

    # Reward for pole being upright
    pole_upright_reward_temp = 10.0 # adjusted to improve sensitivity
    pole_upright_reward = np.exp(-pole_upright_reward_temp * np.abs(pole_angle))

    # Reward for keeping the cart centered
    cart_centered_reward_temp = 2.0 # adjusted to increase contribution
    cart_centered_reward = np.exp(-cart_centered_reward_temp * np.abs(cart_position))

    # Expanding the pole stability reward (previously highly similar values)
    pole_stability_reward_temp = 0.5 # adjusted to widen influence
    pole_stability_reward = np.exp(-pole_stability_reward_temp * np.abs(pole_angular_velocity))

    # Cart stability reward also had similar values, adjusting it
    cart_stability_reward_temp = 0.5 # adjusted to widen influence
    cart_stability_reward = np.exp(-cart_stability_reward_temp * np.abs(cart_velocity))

    # Sum up all reward components, weighted to maintain their influence balance
    total_reward = 2.0 * pole_upright_reward + 1.5 * cart_centered_reward + 1.0 * pole_stability_reward + 1.0 * cart_stability_reward

    # Convert reward components to np.float64 for consistency
    reward_components = {
        "pole_upright_reward": np.float64(pole_upright_reward),
        "cart_centered_reward": np.float64(cart_centered_reward),
        "pole_stability_reward": np.float64(pole_stability_reward),
        "cart_stability_reward": np.float64(cart_stability_reward)
    }

    return np.float64(total_reward), terminated, truncated, reward_components
