def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract relevant variables from the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state

    # Define the thresholds for the pole angle and cart position
    angle_threshold = 12 * (np.pi / 180)  # convert 12 degrees to radians
    position_threshold = 2.4  # max cart position before failure

    # Check if the episode is terminated (if the pole falls or cart moves out of bounds)
    terminated = np.abs(pole_angle) > angle_threshold or np.abs(cart_position) > position_threshold

    # No time limitation for truncation
    truncated = False

    # Reward components to encourage staying balanced
    # Reward for pole being close to upright
    pole_upright_reward_temp = 5.0
    pole_upright_reward = np.exp(-pole_upright_reward_temp * np.abs(pole_angle))

    # Reward for keeping the cart near the center
    cart_centered_reward_temp = 1.0
    cart_centered_reward = np.exp(-cart_centered_reward_temp * np.abs(cart_position))

    # Reward for reducing pole angular velocity
    pole_stability_reward_temp = 0.1
    pole_stability_reward = np.exp(-pole_stability_reward_temp * np.abs(pole_angular_velocity))

    # Reward for reducing cart velocity
    cart_stability_reward_temp = 0.1
    cart_stability_reward = np.exp(-cart_stability_reward_temp * np.abs(cart_velocity))

    # Total reward is the sum of all components
    reward = pole_upright_reward + cart_centered_reward + pole_stability_reward + cart_stability_reward

    # Convert reward components to np.float64
    reward_components = {
        "pole_upright_reward": np.float64(pole_upright_reward),
        "cart_centered_reward": np.float64(cart_centered_reward),
        "pole_stability_reward": np.float64(pole_stability_reward),
        "cart_stability_reward": np.float64(cart_stability_reward)
    }

    return np.float64(reward), terminated, truncated, reward_components
