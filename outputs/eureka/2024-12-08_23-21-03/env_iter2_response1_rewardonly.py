def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract relevant variables from the internal state
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Define the thresholds for terminating the episode
    angle_threshold = 12 * (np.pi / 180)  # radians
    position_threshold = 2.4  # cart position

    # Check if the episode is terminated 
    terminated = np.abs(pole_angle) > angle_threshold or np.abs(cart_position) > position_threshold
    truncated = False

    # Pole Upright Reward: Encourage the pole to be near vertical
    pole_upright_reward_temp = 5.0
    pole_upright_reward = np.exp(-pole_upright_reward_temp * np.abs(pole_angle))

    # Cart Centered Reward: Encourage the cart to remain near the center
    cart_centered_reward_temp = 1.0
    cart_centered_reward = np.exp(-cart_centered_reward_temp * np.abs(cart_position))

    # Pole Stability Reward: Discourage angular velocity
    pole_stability_reward_temp = 2.0
    pole_stability_reward = np.exp(-pole_stability_reward_temp * np.abs(pole_angular_velocity))

    # Cart Stability Reward: Discourage cart velocity
    cart_stability_reward_temp = 1.5
    cart_stability_reward = np.exp(-cart_stability_reward_temp * np.abs(cart_velocity))

    # Calculating a balanced total reward
    total_reward = (3.0 * pole_upright_reward 
                    + 1.0 * cart_centered_reward 
                    + 2.0 * pole_stability_reward 
                    + 2.0 * cart_stability_reward)

    # Convert reward components to np.float64 for consistency
    reward_components = {
        "pole_upright_reward": np.float64(pole_upright_reward),
        "cart_centered_reward": np.float64(cart_centered_reward),
        "pole_stability_reward": np.float64(pole_stability_reward),
        "cart_stability_reward": np.float64(cart_stability_reward)
    }

    return np.float64(total_reward), terminated, truncated, reward_components
