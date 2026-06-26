import ast
import glob
import logging

import json
import os
import subprocess
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils.comp_state_info import CompStateInfo


def parse_transition_features(row):
    """Parse the transition features from a row of the DataFrame.
    For now, they are the last 6 columns of the row."""
    transition_features = row.iloc[-6:].to_dict()
    # add the ranking of the successor state
    successor_ranking = parse_rank(row['successor_rank'])
    transition_features['successor_ranking'] = successor_ranking
    return transition_features


def parse_rank(ranking_str):
    """Parse the ranking string into an integer. Ex. [[assume:1, value:19]] -> 19"""
    if pd.isna(ranking_str):
        raise ValueError("Ranking string is NaN or empty.")
    return int(ranking_str.split(':')[-1].strip(']'))


def parse_state_info(row):
    """Parse the state information from a row of the DataFrame."""
    # State
    state_id = row['state']
    state_ranking = parse_rank(row['state_rank'])
    component_states = row['state_comp_states']
    abstract_state = row['abstract_state']
    state_info = CompStateInfo(state_id, state_ranking, component_states, abstract_state)

    return state_info


def parse_successor_info(row):
    """Parse the successor state information from a row of the DataFrame."""
    successor_state_id = row['successor']
    successor_ranking = parse_rank(row['successor_rank'])
    suc_component_states = row['successor_comp_states']
    abstract_successor = row['abstract_successor']
    successor_info = CompStateInfo(successor_state_id, successor_ranking, suc_component_states, abstract_successor)

    return successor_info


def get_instance_data(problem_name, instance_name):
    """Parse the CSV data into a dictionary mapping state-action pairs to successor states.
    """
    data_base_path = f"train_data/{problem_name}"
    data = pd.read_csv(f"{data_base_path}/{instance_name}_features.csv")

    # convert abstract_state,abstract_successor,state_comp_states,successor_comp_states to lists
    data['abstract_state'] = data['abstract_state'].apply(lambda x: x.strip('[]').split(', '))
    data['abstract_successor'] = data['abstract_successor'].apply(lambda x: x.strip('[]').split(', '))
    data['state_comp_states'] = data['state_comp_states'].apply(lambda x: eval(x.strip()))
    data['successor_comp_states'] = data['successor_comp_states'].apply(lambda x: eval(x.strip()))

    # The actions of the RL agent are all the transitions in the data (state-action-successor)
    #uncontrollable_actions = set(data.loc[data['is_controllable_action'] == False, 'action'].unique())
    #controllable_actions = set(data.loc[data['is_controllable_action'] == True, 'action'].unique())
    # load the actions from the all_actions_info.txt file (to ensure we have all actions, even if not present in this scenario)
    uncontrollable_actions = set(pd.read_csv(f"{data_base_path}/all_uncontrollable_actions.csv")['action'])
    controllable_actions = set(pd.read_csv(f"{data_base_path}/all_controllable_actions.csv")['action'])

    # Build the transition dictionary
    transitions = dict(
        zip(
            zip(data['state'], data['action'], data['successor']),
            data.apply(parse_transition_features, axis=1)
        )
    )
    transitions_df = pd.DataFrame(transitions).T

    # Create a mapping from state IDs to CompStateInfo objects
    state_infos = data.apply(parse_state_info, axis=1)
    successor_infos = data.apply(parse_successor_info, axis=1)
    state_id_to_info = {}

    for state_info in pd.concat([state_infos, successor_infos]):
        if state_info.state_id not in state_id_to_info:
            state_id_to_info[state_info.state_id] = state_info
            # add all the transitions for that state in a batch
            state_transitions = transitions_df.loc[
                transitions_df.index.get_level_values(0) == state_info.state_id
                ]
            state_info.add_transitions_batch(state_transitions)

    # If the first state is fully uncontrollable, we need to add a "Tau" action to it
    first_state_info = state_id_to_info[0]
    if first_state_info.is_fully_uncontrollable():
        controllable_actions.add('Tau')
        first_state_info.add_transition((0, 'Tau', 0), controllable=True)

    # Get all roles in the system
    all_roles = set()
    for column in ['abstract_successor', 'abstract_state']:
        all_roles.update(data[column].explode())

    # Get the range in values of component states for each component
    state_comp_states_df = pd.DataFrame(data['state_comp_states'].tolist())
    successor_comp_states_df = pd.DataFrame(data['successor_comp_states'].tolist())

    combined_df = pd.concat([state_comp_states_df, successor_comp_states_df])
    # number of states = max state index + 1
    component_number_of_states = [s_max + 1 for s_max in combined_df.max(axis=0).tolist()]

    # TODO we are returning too much variables, maybe it's better to return a well structured df?
    return (state_id_to_info, transitions_df, uncontrollable_actions, controllable_actions,
            all_roles, component_number_of_states)


def run_problem_instance(instance_filepath, output_path, output_filename, controller_name, extra_options="",
                         timeout=120, jarname="mtsa-1.0-SNAPSHOT.jar" ):
    """
    Run the problem instance with the given parameters
    :param instance_filepath: Path to the instance file, -i option in LTSABatch
    :param output_filename:
    :param output_path: Path to store the results, must exist, -o option in LTSABatch
    :param controller_name: Name of the controller to generate, -c option in LTSABatch
    :param extra_options: String with extra options to pass to the ltsa.ui.LTSABatch command
    :param timeout: Timeout in seconds
    :return:
    """
    os.makedirs(output_path, exist_ok=True)

    # Normalize paths for the OS (convert to forward slashes for cross-platform compatibility)
    instance_filepath = instance_filepath.replace('\\', '/')
    output_path = output_path.replace('\\', '/')
    output_filepath = f"{output_path}/{output_filename}"

    memory = 8  # 8 GB
    mtsa_command = (f"java -Xmx{memory}g -cp ./{jarname} ltsa.ui.LTSABatch "
                    f"-i {instance_filepath} -c {controller_name} "
                    f"-o {output_filepath} {extra_options}")
    print("\033[93m" + f"Running command: {mtsa_command}" + "\033[0m")
    process = subprocess.Popen(mtsa_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    result = 'OK'
    start_time = time.time()
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        result = 'TO'
        return timeout, result, "", ""
    end_time = time.time()

    stdout_decoded = stdout.decode(errors='replace') if stdout else ""
    stderr_decoded = stderr.decode(errors='replace') if stderr else ""

    if 'Invalid option' in stdout_decoded:
        result = 'Invalid option'

    if process.returncode != 0:
        if "java.lang.OutOfMemoryError" in stderr_decoded:
            result = 'OOM'
        elif "No controller" in stderr_decoded:
            result = 'No controller'
        else:
            raise RuntimeError(f"Command failed with return code {process.returncode}: {stderr_decoded}")

    execution_time = end_time - start_time
    return execution_time, result, stdout_decoded, stderr_decoded


def save_json(data, path):  # TODO cambiarlo para guardar df
    # Convert numpy.int64 to int
    def convert_to_serializable(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Save the metrics to a file
    with open(path, "w") as f:
        json.dump(data, f, default=convert_to_serializable, indent=4)


def show_histogram(y_test, y_pred):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot the histogram for predicted rankings
    axs[0].hist(y_pred, bins=20, color='blue', alpha=0.7)
    axs[0].set_title('Predicted Rankings')
    axs[0].set_xlabel('Ranking')
    axs[0].set_ylabel('Frequency')

    # Plot the histogram for real rankings
    axs[1].hist(y_test, bins=20, color='green', alpha=0.7)
    axs[1].set_title('Real Rankings')
    axs[1].set_xlabel('Ranking')

    plt.tight_layout()
    # plt.show()


def plot_learning_curve(model_name, results_path, rewards, window=100):
    # Input validation
    if not rewards or len(rewards) == 0:
        raise ValueError("Rewards list is empty or invalid.")

    # Compute moving average
    moving_avg = []
    for i in range(len(rewards)):
        window_start = max(0, i - window + 1)  # Inclusive start index
        window_data = rewards[window_start:i + 1]
        moving_avg.append(np.mean(window_data))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward per Episode", color="blue")
    plt.plot(moving_avg, label=f"Moving Average (window={window})", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Learning Curve {model_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    ymin = min(rewards)
    plt.ylim(ymin - 0.2, max(max(rewards), 1) + 0.1)

    # Ensure directory exists
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(f"{results_path}/{model_name}_learning_curve.png")
    # plt.show()


def plot_winning_rate(model_name, results_path, wins, window=100):
    moving_avg = [np.mean(wins[max(0, i - window):i + 1]) for i in range(len(wins))]
    plt.figure(figsize=(10, 6))
    plt.plot(wins, label="Win Rate per Episode")
    plt.plot(moving_avg, label=f"Moving Average (window={window})", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title(f"Winning Rate {model_name}")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.savefig(f"{results_path}/{model_name}_winning_rate.png")
    # plt.show()


def plot_loss_curve(model_name, results_path, episode_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_losses, label="Average Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title(f"Loss Curve {model_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{results_path}/{model_name}_loss_curve.png")
    # plt.show()


def save_results(episodes_info, env, results_path, model_name):
    # save the episodes info to a csv file
    episodes_info.to_csv(f"{results_path}/{model_name}_episodes_info.csv", index_label='episode')

    # save the composition as a graphml file
    # nx.write_graphml(env.composition, f"{results_path}/composition.graphml")  # TODO fix


def discover_scenarios(base_dir, problem):
    """Discover all scenarios from LTS files in the problem directory."""
    problem_dir = f"{base_dir}/{problem}"
    scenarios = []

    if not os.path.exists(problem_dir):
        print(f"Warning: Directory {problem_dir} not found")
        return scenarios

    # Find all .lts files
    lts_files = glob.glob(os.path.join(problem_dir, f"{problem}_*.lts"))

    for lts_file in lts_files:
        filename = os.path.basename(lts_file)
        # Extract scenario part: from "sokoban-4-1_1obstacles_1.lts" get "1obstacles_1"
        # Remove the problem prefix and .lts suffix
        scenario = filename.replace(f"{problem}_", "").replace(".lts", "")
        scenarios.append(scenario)

    # Sort scenarios for consistent ordering
    scenarios.sort()
    return scenarios
