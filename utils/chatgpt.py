from numpy import append, isin
from openai import OpenAI
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam, ChatCompletionContentPartParam
import base64

import os

file_location = os.path.abspath(__file__)

parent_directory = os.path.dirname(os.path.dirname(file_location))

with open(os.path.join(parent_directory, "openai_key.txt")) as f:
    api_key = f.read()

client = OpenAI(api_key=api_key)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_completions(
    model, messages: Iterable[ChatCompletionMessageParam], temperature, n
):
    completions = client.chat.completions.create(
        model=model, messages=messages, n=n, temperature=temperature
    )

    return completions

def create_message() -> ChatCompletionUserMessageParam:
    return {
        "role": "user",
        "content": [],
    }

def append_image(message: ChatCompletionUserMessageParam, image_path:str) -> ChatCompletionUserMessageParam:
    if isinstance(message['content'], str):
        message['content'] = [{"type": "text", "text": message['content']}]
    assert isinstance(message['content'], list), 'bug!'
    base64_image = encode_image(image_path)
    message['content'].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}) # type: ignore
    return message

def append_text(message: ChatCompletionUserMessageParam, text:str) -> ChatCompletionUserMessageParam:
    if isinstance(message['content'], str):
        message['content'] = [{"type": "text", "text": message['content']}]
    assert isinstance(message['content'], list), 'bug!'
    message['content'].append({"type": "text", "text": text})
    return message

if __name__ == "__main__":
    messages = [
            {
                "role": "system",
                "content": 'You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.\nYour goal is to write a reward function for the environment that will help the agent learn the task described in text. \nYour reward function should use useful variables from the environment as inputs. As an example,\nthe reward function signature can be: def compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:\n    ...\n    return reward, terminated, truncated, reward_components\n.\nPlease make sure that the reward or its components are returned as np.float64, rather than float, int, np.ndarray, or any other type.The output of the reward function should consist of four items:\n    (1) the total reward,\n    (2) whether the episode is terminated,\n    (3) whether the episode is truncated, and\n    (4) a dictionary of each individual reward component.\nThe code output should be formatted as a python code string: "```python ... ```".\nSome helpful tips for writing the reward function code:\n    (1) You may find it helpful to normalize the reward to a fixed range by applying transformations like torch.exp to the overall reward or its components\n    (2) If you choose to transform a reward component, then you must also introduce a temperature parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable\n    (3) Use 4 spaces for indentation.\n    (5) Most importantly, you should assume that the function does not belong to a class, so you should not include the "self" parameter in the function signature. Only the values provided in the "internal_state" variable will be available to you.\n',
            },
            {
                "role": "user",
                "content": "The Python environment is import gymnasium as gym\nfrom gymnasium.envs.classic_control.cartpole import CartPoleEnv\nfrom gymnasium import spaces\nimport numpy as np\n\nclass CustomCartPoleEnv(CartPoleEnv):\n    def get_state(self) -> np.ndarray:\n        # Compute the observation\n        # swing threshold is just a constant that is added to the state for the ease of access.\n        return np.array([self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity, self.swing_threshold, 0.0 if self.has_touched_left_threshold else 1.0, 0.0 if self.has_touched_right_threshold else 1.0])\n    . Write a reward function for the following task: to balance a pole on a cart so that both of the following are satisfied: (1) the pole stays upright and (2) the cart can move back and forth periodically from position of -swing_threashold to swing_threshold, as fast as possible..",
            },
            {
                "role": "assistant",
                "content": '```python\ndef compute_reward(internal_state: np.ndarray) -> Tuple[np.float64, bool, bool, Dict[str, np.float64]]:\n    # Extract state variables\n    cart_position = internal_state[0]\n    cart_velocity = internal_state[1]\n    pole_angle = internal_state[2]\n    pole_angular_velocity = internal_state[3]\n    swing_threshold = internal_state[4]\n    has_touched_left_threshold = internal_state[5]\n    has_touched_right_threshold = internal_state[6]\n\n    # Define constants\n    pole_upright_threshold = 0.05  # radians, small angle for upright position\n    position_threshold = swing_threshold  # cart should move between -swing_threshold and swing_threshold\n    angle_penalty_temperature = 10.0\n    velocity_reward_temperature = 0.1\n\n    # Reward components\n    # 1. Pole uprightness reward\n    pole_upright_reward = np.exp(-angle_penalty_temperature * abs(pole_angle))\n\n    # 2. Cart movement reward\n    if has_touched_left_threshold and has_touched_right_threshold:\n        cart_movement_reward = np.exp(velocity_reward_temperature * abs(cart_velocity))\n    else:\n        cart_movement_reward = 0.0\n\n    # Total reward\n    total_reward = np.float64(pole_upright_reward + cart_movement_reward)\n\n    # Check termination conditions\n    terminated = bool(abs(pole_angle) > pole_upright_threshold or abs(cart_position) > position_threshold)\n    truncated = False  # Assuming no truncation condition is specified\n\n    # Reward components dictionary\n    reward_components = {\n        "pole_upright_reward": np.float64(pole_upright_reward),\n        "cart_movement_reward": np.float64(cart_movement_reward)\n    }\n\n    return total_reward, terminated, truncated, reward_components\n```',
            },
            {
                "role": "user",
                "content": "We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every 5000 time steps and the maximum, mean, minimum values encountered:\npole_upright_reward: ['185.53', '172.03', '236.70', '338.63', '424.32', '402.27', '322.19', '482.86', '401.98', '342.37', '213.86', '395.17', '400.49', '482.66', '425.63', '414.10', '427.63', '484.43', '484.12', '364.78', '448.63', '484.17', '484.78', '484.75', '484.97', '475.92', '481.34', '485.02', '484.92', '480.47', '484.96', '485.14', '485.15', '484.97', '484.06', '485.26', '485.03', '485.30', '484.46', '323.67', '466.03', '435.27', '474.50', '483.32', '462.79', '467.16', '484.17', '483.57', '323.41', '484.75', '484.46', '484.81', '485.40', '484.94', '485.08', '483.98', '485.11', '485.10', '485.04', '262.98'], Max: 485.89, Mean: 434.61, Min: 0.61 \ncart_movement_reward: ['205.04', '189.68', '257.40', '361.12', '446.38', '422.66', '343.49', '505.66', '421.80', '361.59', '234.98', '416.33', '420.54', '505.76', '445.86', '436.61', '448.32', '505.62', '505.80', '384.94', '470.42', '505.29', '505.63', '505.69', '505.57', '495.89', '504.48', '505.30', '505.49', '501.52', '505.53', '505.79', '505.59', '505.76', '505.28', '505.49', '505.35', '505.62', '505.97', '337.09', '486.69', '456.73', '496.34', '506.30', '484.70', '489.67', '505.88', '505.95', '337.49', '505.65', '505.76', '505.27', '505.36', '505.68', '505.23', '506.12', '505.41', '505.42', '505.37', '292.44'], Max: 506.98, Mean: 455.50, Min: 1.02 \nReward from generated code: ['390.57', '361.71', '494.10', '699.75', '870.70', '824.93', '665.67', '988.52', '823.78', '703.96', '448.84', '811.50', '821.03', '988.41', '871.49', '850.71', '875.96', '990.05', '989.92', '749.73', '919.05', '989.46', '990.40', '990.43', '990.54', '971.81', '985.82', '990.32', '990.42', '981.99', '990.48', '990.93', '990.74', '990.73', '989.34', '990.75', '990.39', '990.92', '990.43', '660.76', '952.72', '892.01', '970.83', '989.62', '947.48', '956.82', '990.05', '989.52', '660.90', '990.40', '990.22', '990.08', '990.77', '990.62', '990.31', '990.10', '990.52', '990.52', '990.41', '555.42'], Max: 991.56, Mean: 890.11, Min: 1.63 \nground-truth reward: ['202.67', '187.33', '252.00', '355.67', '440.67', '417.00', '338.00', '500.00', '416.33', '356.33', '229.67', '410.67', '415.00', '500.00', '440.33', '431.33', '442.67', '500.00', '500.00', '379.33', '464.67', '500.00', '500.00', '500.00', '500.00', '490.67', '498.00', '500.00', '500.00', '496.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '333.67', '480.67', '450.67', '490.33', '500.00', '478.67', '483.33', '500.00', '500.00', '333.67', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '286.33'], Max: 500.00, Mean: 450.03, Min: 1.00 \nepisode_length: ['202.67', '187.33', '252.00', '355.67', '440.67', '417.00', '338.00', '500.00', '416.33', '356.33', '229.67', '410.67', '415.00', '500.00', '440.33', '431.33', '442.67', '500.00', '500.00', '379.33', '464.67', '500.00', '500.00', '500.00', '500.00', '490.67', '498.00', '500.00', '500.00', '496.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '333.67', '480.67', '450.67', '490.33', '500.00', '478.67', '483.33', '500.00', '500.00', '333.67', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '500.00', '286.33'], Max: 500.00, Mean: 450.03, Min: 1.00 \nPlease carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:\n    (1) If the success rates are always near zero, then you must rewrite the entire reward function\n    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider\n        (a) Changing its scale or the value of its temperature parameter\n        (b) Re-writing the reward component \n        (c) Discarding the reward component\n    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range\nPlease analyze each existing reward component in the suggested manner above first, and then write the reward function code. The output of the reward function should consist of four items:\n    (1) the total reward,\n    (2) whether the episode is terminated,\n    (3) whether the episode is truncated, and\n    (4) a dictionary of each individual reward component.\nThe code output should be formatted as a python code string: \"```python ... ```\".\nSome helpful tips for writing the reward function code:\n    (1) You may find it helpful to normalize the reward to a fixed range by applying transformations like torch.exp to the overall reward or its components\n    (2) If you choose to transform a reward component, then you must also introduce a temperature parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable\n    (3) Use 4 spaces for indentation.\n    (5) Most importantly, you should assume that the function does not belong to a class, so you should not include the \"self\" parameter in the function signature. Only the values provided in the \"internal_state\" variable will be available to you.\n",
            },
        ]
    append_text(messages[-1], "In addition to the data, we have also attached screenshots of several frames from a sample rollout, arranged in the order they were captured, for your reference. When drafting the new reward function, please analyze the agent’s behavior and address the issues in the current reward function as illustrated by the agent’s actions in the screenshots.")
    append_image(messages[-1], "/Users/chunhaozou/Desktop/drl/episode_1_frame_0.png")
    append_image(messages[-1], "/Users/chunhaozou/Desktop/drl/episode_1_frame_7.png")
    append_image(messages[-1], "/Users/chunhaozou/Desktop/drl/episode_1_frame_14.png")
    append_image(messages[-1], "/Users/chunhaozou/Desktop/drl/episode_1_frame_21.png")
    append_image(messages[-1], "/Users/chunhaozou/Desktop/drl/episode_1_frame_28.png")
    print(messages)
    completions = get_completions(
        "gpt-4o-mini",
        messages,
        0.7,
        1,
    )

    


    for completion in completions.choices:
        print(completion.message.content)
