def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:
    # Extract the specific state variables for ease of use
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = internal_state
    
    # Define thresholds and weights for each component
    angle_threshold = np.radians(15)  # Angle threshold in radians (equivalent to +-15 degrees)
    position_threshold = 2.4  # Cart position threshold
    
    # Reward component for maintaining a low pole angle
    angle_temp = 5.0
    if abs(pole_angle) <= angle_threshold:
        angle_reward = np.exp(-angle_temp * (pole_angle / angle_threshold)**2)
    else:
        angle_reward = 0.0

    # Punish for the cart moving too far from the center
    position_temp = 5.0
    if abs(cart_position) <= position_threshold:
        position_reward = np.exp(-position_temp * (cart_position / position_threshold)**2)
    else:
        position_reward = 0.0

    # Encourage low velocities
    velocity_temp = 0.1
    velocity_reward = np.exp(-velocity_temp * (cart_velocity**2 + pole_angular_velocity**2))

    # Total reward is some weighted combination of these components
    total_reward = (angle_reward + position_reward + velocity_reward) / 3.0
    
    # Episode terminates if the pole angle or cart position goes out of the defined boundaries
    terminated = abs(pole_angle) > angle_threshold or abs(cart_position) > position_threshold
    truncated = False  # Truncation logic depends on other factors like maximum step count, not state
    
    # Collect individual reward components
    reward_components = {
        "angle_reward": np.float64(angle_reward),
        "position_reward": np.float64(position_reward),
        "velocity_reward": np.float64(velocity_reward)
    }
    
    return np.float64(total_reward), terminated, truncated, reward_components
