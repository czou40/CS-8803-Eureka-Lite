from ast import arg
import code
from typing import cast
from cv2 import log
import hydra
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
import re
import subprocess
from pathlib import Path
import shutil
import time

from utils.log_utils import pretty_print_yellow
from utils.misc import *
from utils.file_utils import find_screenshots
from utils.extract_task_code import *
from utils.chatgpt import get_completions, ChatCompletionMessageParam, append_text, append_image

EUREKA_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

SKIP_GENERATION = True


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()

    task_file = f"{EUREKA_ROOT_DIR}/envs/{env_name}.py"
    task_obs_file = f"{EUREKA_ROOT_DIR}/envs/{env_name}_obs.py"
    task_train_file = f"{EUREKA_ROOT_DIR}/envs/{env_name}_train_eval.py"
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{EUREKA_ROOT_DIR}/out.py"

    # Loading all text prompts
    prompt_dir = f"{EUREKA_ROOT_DIR}/utils/prompts"
    initial_system = file_to_string(f"{prompt_dir}/initial_system.txt")
    code_output_tip = file_to_string(f"{prompt_dir}/code_output_tip.txt")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    initial_user = file_to_string(f"{prompt_dir}/initial_user.txt")
    reward_signature = file_to_string(f"{prompt_dir}/reward_signature.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    screnshot_feedback = file_to_string(f"{prompt_dir}/screenshot_feedback.txt")
    execution_error_feedback = file_to_string(
        f"{prompt_dir}/execution_error_feedback.txt"
    )

    initial_system = (
        initial_system.format(task_reward_signature_string=reward_signature)
        + code_output_tip
    )
    initial_user = initial_user.format(
        task_obs_code_string=task_obs_code_string, task_description=task_description
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]

    # task_code_string = task_code_string.replace(task, task+suffix)

    # TODO: remove this part because we no longer use isaacgym
    # # Create Task YAML files
    # create_task(EUREKA_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.0
    max_successes = []
    average_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    best_data_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None
    max_reward_model_data_path = None

    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        prompt_tokens = 0
        chunk_size = cfg.sample

        logging.info(
            f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}"
        )

        logging.info(json.dumps(messages, indent=4))

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = get_completions(
                        model, messages, cfg.temperature, chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            usage = response_cur.usage
            assert usage
            prompt_tokens = usage.prompt_tokens
            total_completion_token += usage.completion_tokens
            total_token += usage.total_tokens

        if cfg.sample == 1:
            logging.info(
                f"Iteration {iter}: GPT Output:\n "
                + responses[0].message.content
                + "\n"
            )

        # Logging Token Information
        logging.info(
            f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
        )

        code_runs = []
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id].message.content
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r"```python(.*?)```",
                r"```(.*?)```",
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]

            code_string = None

            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            assert isinstance(code_string, str), "Code string is not a string!"

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])

            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(
                    f"Iteration {iter}: Code Run {response_id} cannot parse function signature!"
                )
                continue

            code_runs.append(code_string)
            # reward_signature = [
            #     f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
            #     f"self.extras['gpt_reward'] = self.rew_buf.mean()",
            #     f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
            # ]
            reward_signature = [
                f"self.reward, self.terminated, self.truncated, self.rew_dict = {gpt_reward_signature}",
            ]
            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])
            # pretty_print_yellow(f'reward_signature: {reward_signature}')
            # pretty_print_yellow(f'task_code_string: {task_code_string}')
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace(
                    "def compute_reward(self):",
                    "def compute_reward(self):\n" + reward_signature,
                )
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace(
                    "def compute_reward(self, actions):",
                    "def compute_reward(self, actions):\n" + reward_signature,
                )
            else:
                raise NotImplementedError

            # Save the new environment code when the output contains valid code string!
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(
                output_file,
                "w",
            ) as file:
                file.writelines(task_code_string_iter + "\n")
                file.writelines("from typing import Tuple, Dict" + "\n")
                file.writelines("import math" + "\n")
                file.writelines("import numpy as np" + "\n")
                file.writelines("from numpy import ndarray, float64" + "\n")
                file.writelines(code_string + "\n")

            with open(
                f"env_iter{iter}_response{response_id}_rewardonly.py", "w"
            ) as file:
                file.writelines(code_string + "\n")

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            pretty_print_yellow(f"Generated code saved to {output_file}")
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, "w") as f:
                process = subprocess.Popen(
                    [
                        "python",
                        "-u",
                        task_train_file,
                        "--iter",
                        str(iter),
                        "--response_id",
                        str(response_id),
                        "--seed",
                        str(0),
                        "--total_timesteps",
                        str(cfg.total_timesteps),
                        "--is_eval",
                        "false",
                        "--log_timestep_freq",
                        str(cfg.log_timestep_freq),
                        "--eval_num_episodes",
                        str(cfg.num_eval),
                    ],
                    stdout=f,
                    stderr=f,
                )
            # block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)

        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []
        data_paths = []

        exec_success = False
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            pretty_print_yellow(f"RL run {response_id} completed")
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            npz_log_file_path = f"env_iter{iter}_response{response_id}.npz"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            data_paths.append(npz_log_file_path)
            try:
                with open(rl_filepath, "r") as f:
                    stdout_str = f.read()
            except:
                content = execution_error_feedback.format(
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                )
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ""
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == "":
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                # lines = stdout_str.split("\n")
                # assert len(lines) > 0, "No output from the RL training!"
                # line = ""
                # for i, line in enumerate(lines):
                #     if line.startswith("Tensorboard Directory:"):
                #         break
                # assert line != "", "No tensorboard log directory found!"
                # tensorboard_logdir = line.split(":")[-1].strip()
                
                assert os.path.exists(npz_log_file_path), "No npz log file found!"
                logged_data = np.load(npz_log_file_path)
                # tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                # max_iterations = np.array(logged_data["gt_reward"]).shape[0]
                # epoch_freq = max(int(max_iterations // 10), 1)

                content += policy_feedback.format(time_step_freq=cfg.log_timestep_freq)

                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in logged_data and "gpt_reward" in logged_data:
                    gt_reward = np.array(logged_data["gt_reward"])
                    gpt_reward = np.array(logged_data["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in logged_data:
                    if not metric.startswith("max_mean_min_") and not metric.startswith("last_round_"):
                        metric_cur = [
                            "{:.2f}".format(x)
                            for x in logged_data[metric]
                        ]
                        metric_cur_max = logged_data["max_mean_min_" + metric][0]
                        metric_cur_mean = logged_data["max_mean_min_" + metric][1]
                        metric_cur_min =  logged_data["max_mean_min_" + metric][2]

                        # Modifed this part: if there is no consecutive_successes, provide ground-truth score
                        if "consecutive_successes" == metric or "consecutive_successes" not in logged_data and "gt_reward" == metric:
                            successes.append(metric_cur_max)

                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in logged_data:
                                if metric == "gt_reward":
                                    metric_name = "ground-truth reward"
                                else:
                                    metric_name = "Reward from generated code"
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.0)
            max_successes.append(DUMMY_FAILURE)
            average_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            best_data_paths.append(None)
            logging.info(
                "All code generation failed! Repeat this iteration from the current message checkpoint!"
            )
            continue

        assert len(successes) == len(contents), "Bug: Successes and contents mismatch!"
        # Select the best code sample based on the success rate or best ground-truth reward
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.0) / cfg.sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]
            max_reward_model_data = data_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        average_successes.append(np.mean(successes))
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])
        best_data_paths.append(data_paths[best_sample_idx])

        logging.info(
            f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}"
        )
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(
            f"Iteration {iter}: GPT Output Content:\n"
            + responses[best_sample_idx].message.content
            + "\n"
        )
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(8, 16))
        fig.suptitle(f"{cfg.env.task}")

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].plot(x_axis, np.array(average_successes))
        axs[0].set_title("Average and Max Scores")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig("summary.png")
        np.savez(
            "summary.npz",
            max_successes=max_successes,
            average_successes=average_successes,
            raw_scores = successes,
            execute_rates=execute_rates,
            best_code_paths=best_code_paths,
            max_successes_reward_correlation=max_successes_reward_correlation,
        )

        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx].message.content}]  # type: ignore
            messages += [{"role": "user", "content": best_content}]  # type: ignore
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx].message.content}  # type: ignore
            messages[-1] = {"role": "user", "content": best_content}  # type: ignore

        if cfg.use_screenshots:
            logging.info(f"Iteration {iter}: Using Screenshots")
            # find screenshot files
            screenshot_files = find_screenshots(iter, int(best_sample_idx))

            if screenshot_files:
                logging.info(f"Found {len(screenshot_files)} screenshots: {screenshot_files}")
                append_text(messages[-1], screnshot_feedback)  # type: ignore
                for screenshot_file in screenshot_files:
                    append_image(messages[-1], screenshot_file) # type: ignore
            else:

                logging.info(f"No screenshots found for iteration {iter}, response {best_sample_idx}")
        else:
            logging.info(f"Iteration {iter}: Do not use Screenshots")

        # Save dictionary as JSON file
        with open("messages.json", "w") as file:
            json.dump(messages, file, indent=4)

    # Evaluate the best reward code many times
    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info(
            "Please double check the output env_iter*_response*.txt files for repeating errors!"
        )
        exit()
    logging.info(
        f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}"
    )
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")

    # Modification: we have already saved the results, so no need to run again. Just read from the data
    # shutil.copy(max_reward_code_path, output_file)

    # eval_runs = []
    # for i in range(cfg.num_eval):

    #     # Execute the python file with flags
    #     rl_filepath = f"reward_code_eval{i}.txt"
    #     with open(rl_filepath, "w") as f:
    #         # process = subprocess.Popen(['python', '-u', task_train_file,
    #         #                             'hydra/output=subprocess',
    #         #                             f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
    #         #                             f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
    #         #                             f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
    #         #                             ],
    #         #                             stdout=f, stderr=f)
    #         process = subprocess.Popen(["python", "-u", task_train_file])

    #     # block_until_training(rl_filepath)
    #     eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i in range(cfg.num_eval):
        # rl_filepath = f"reward_code_eval{i}.txt"
        # with open(rl_filepath, "r") as f:
        #     stdout_str = f.read()
        # lines = stdout_str.split("\n")
        # line = ""
        # for i, line in enumerate(lines):
        #     if line.startswith("Tensorboard Directory:"):
        #         break
        # assert line != "", "No tensorboard log directory found!"
        # tensorboard_logdir = line.split(":")[-1].strip()

        logged_data = logged_data = np.load(best_data_paths[-1])
        if "consecutive_successes" in logged_data:
            max_success = max(logged_data["last_round_consecutive_successes"])
        else:
            assert "gt_reward" in logged_data
            max_success = max(logged_data["last_round_gt_reward"])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in logged_data and "gpt_reward" in logged_data:
            gt_reward = np.array(logged_data["last_round_gt_reward"])
            gpt_reward = np.array(logged_data["last_round_gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(
        f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}"
    )
    logging.info(
        f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}"
    )
    np.savez(
        "final_eval.npz",
        reward_code_final_successes=reward_code_final_successes,
        reward_code_correlations_final=reward_code_correlations_final,
    )


if __name__ == "__main__":
    main()
