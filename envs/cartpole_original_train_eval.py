from collections import defaultdict
from email.policy import default
import os
from pydoc import render_doc
import re
import gymnasium as gym
from pkg_resources import parse_requirements
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# from out import CustomCartPoleEnv
import argparse
import sys

import numpy as np

import importlib.util


def dynamic_import_custom_class(file_name, env_name):
    # Ensure the file exists
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File {file_name} does not exist.")

    # Get the module name by stripping the path and extension
    module_name = os.path.splitext(os.path.basename(file_name))[0]

    # Load the module from the given file
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the CustomCartPoleEnv from the loaded module
    if not hasattr(module, 'CustomCartPoleEnv'):
        raise AttributeError(f"'CustomCartPoleEnv' not found in {file_name}.")
    return getattr(module, 'CustomCartPoleEnv')

parser = argparse.ArgumentParser(description="Training script with various parameters.")
# Define arguments
parser.add_argument("--iter", type=int, default=0, help="Iteration (default: 0)")
parser.add_argument("--response_id", type=int, default=0, help="Response ID (default: 0)")
parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
parser.add_argument("--capture_video", type=lambda x: (str(x).lower() == 'true'), default=False, 
                    help="Whether to capture video (True/False, default: False)") # not used
parser.add_argument("--total_timesteps", type=int, default=100000, help="Max number of iterations (default: 100)")
parser.add_argument("--log_timestep_freq", type=int, default=5000, help="Log timestep frequency (default: 5000)")
parser.add_argument("--is_eval", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--eval_num_episodes", type=int, default=5, help="Number of episodes to evaluate the trained model (default: 5)")

# Parse arguments
args = parser.parse_args()

output_model_name = f"model_{args.iter}_{args.response_id}"
env_file_path = f"env_iter{args.iter}_response{args.response_id}.py"
npz_log_file_path = f"env_iter{args.iter}_response{args.response_id}.npz"

# Load the custom environment
CustomCartPoleEnv = dynamic_import_custom_class(f"{env_file_path}", "CustomCartPoleEnv")

# # Redirect all prints to the output_file_path
# f = open(output_file_path, 'w')
# sys.stdout = f

def main():
    eval_num_episodes:int = args.eval_num_episodes
    if not args.is_eval:
        # Create the custom environment
        env = CustomCartPoleEnv()
        # # Wrap in DummyVecEnv for training
        # vec_env = DummyVecEnv([lambda: env])

        # Create the PPO model
        model = PPO("MlpPolicy", env, verbose=1)

        total_timesteps_left:int = args.total_timesteps
        log_freq:int = args.log_timestep_freq

        logged_data: defaultdict[str, list[np.float64]] = defaultdict(list)
        logged_data_max: dict[str, np.float64] = {}
        logged_data_min: dict[str, np.float64] = {}
        
        current_time_step_logged_data: defaultdict[str, list[np.float64]] = defaultdict(list)

        while total_timesteps_left > 0:
            # Train the model
            model.learn(total_timesteps=log_freq if total_timesteps_left >= log_freq else total_timesteps_left, log_interval=10)
            total_timesteps_left -= log_freq
            current_time_step_logged_data: defaultdict[str, list[np.float64]] = defaultdict(list)

            for i in range(eval_num_episodes):
                obs, _ = env.reset()
                done, truncated = False, False
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, truncated, _ = env.step(action)
                current_trial_log_summary = env.get_log_summary()    
                print(f"Episode {i+1} finished with reward:", current_trial_log_summary)            
                for key, value in current_trial_log_summary.items():
                    current_time_step_logged_data[key].append(value)

            current_time_step_logged_data_mean = {key: np.array(value).mean() for key, value in current_time_step_logged_data.items()}
            current_time_step_logged_data_max = {key: np.array(value).max() for key, value in current_time_step_logged_data.items()}
            current_time_step_logged_data_min = {key: np.array(value).min() for key, value in current_time_step_logged_data.items()}
            for key, value in current_time_step_logged_data_mean.items():
                logged_data[key].append(value)

            if not logged_data_max:
                logged_data_max = current_time_step_logged_data_max
                logged_data_min = current_time_step_logged_data_min
            else:
                for key in current_time_step_logged_data_max.keys():
                    logged_data_max[key] = max(logged_data_max[key], current_time_step_logged_data_max[key])
                    logged_data_min[key] = min(logged_data_min[key], current_time_step_logged_data_min[key])
        # Save the model
        model.save(output_model_name)
        print("Model saved!")

        # # Save the logs as npz
        logged_data_npz = {key: np.array(value) for key, value in logged_data.items()}
        for key, value in logged_data.items():
            logged_data_npz[key] = np.array(value)
            logged_data_npz['max_mean_min_'+key] = np.array([logged_data_max[key], logged_data_npz[key].mean(), logged_data_min[key]])
            logged_data_npz['last_round_'+key] = np.array(current_time_step_logged_data[key])
        np.savez(npz_log_file_path, **logged_data_npz)
        print("Logs saved!")

    else:
        # # Test the trained model
        # test_env = CustomCartPoleEnv()
        # # Create the PPO model
        # model = PPO.load(output_model_name, env=test_env)
        # print("Evaluate the trained model...")
        # for i in range(eval_num_episodes):
        #     obs, info = test_env.reset()
        #     done, truncated = False, False
        #     total_rewards = 0
        #     while not (done or truncated):
        #         action, _ = model.predict(obs, deterministic=True)
        #         obs, reward, done, truncated, info = test_env.step(action)
        #         total_rewards += reward
        #     print(f"Episode {i+1} finished with reward:", total_rewards)

        # test_env.close()
        pass # this part is deprecated

if __name__ == "__main__":
    print('name:', __name__)
    main()

    # # visualize
    # # Note: The visualization (env.render()) typically needs to run in an environment
    # # that supports GUI rendering. The prints here will still be redirected to the file.
    # env = CustomCartPoleEnv(render_mode='human')
    # model = PPO.load(output_model_name, env=env)
    
    # obs, info = env.reset()
    # num_episodes = 0
    # while True:
    #     action, _ = model.predict(obs, deterministic=True)
    #     print(env.time_step)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     env.render()
    #     if terminated or truncated:
    #         obs, info = env.reset()
    #         num_episodes += 1
    #         if num_episodes >= 5:
    #             break
    # env.close()

# f.close()  # Close the file when done
