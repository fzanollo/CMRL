import os

import pandas as pd

from utils.utils import run_problem_instance

BASE_INPUT = "train_data/lts"
TIMEOUT = 60 * 30

def move_file_with_replacement(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.rename(src, dst)

def run_monolithic_problem(base_input, problem, instances, skip_existing=False):
    print(f"Running monolithic for problem {problem}")
    problem_results = []
    for instance_name in instances:
        instance_filepath = f"{base_input}/{problem}/{instance_name}.lts"
        output_path = f"train_data/{problem}"
        os.makedirs(output_path, exist_ok=True)

        if skip_existing and os.path.exists(f"{output_path}/{instance_name}_features.csv"):
            print(f"Skipping {instance_name}. Features file already exists.")
            continue

        output_filename = f"{instance_name}_controller.fsp"
        extra_options = "--outputFeatureData"
        execution_time, result, stdout_decoded, stderr_decoded = run_problem_instance(instance_filepath, output_path,
                                                                                      output_filename,
                                               "MonolithicController", extra_options, timeout=TIMEOUT)

        if result == 'OK' or result == 'No controller':
            # Move the generated files to the output path
            for file_suffix in ["_features.csv", "_subgraph.fsp"]:
                src_file = f"{base_input}/{problem}/{instance_name}{file_suffix}"
                dst_file = f"{output_path}/{instance_name}{file_suffix}"
                move_file_with_replacement(src_file, dst_file)

            print(f"Processed {instance_name} in {execution_time:.2f}s. Result: {result}.")
        elif result == 'TO':
            print(f"Timeout processing {instance_name}")
        else:
            print(f"Error processing {instance_name}. Result: {result}")

        data = (instance_name, execution_time, result) + parse_stderr(stderr_decoded)
        problem_results.append(data)

    results_df = pd.DataFrame(problem_results,
                              columns=["instance_name", "execution_time", "result",
                                       "n_transitions", "max_states", "winning_state_size", "game_state_size"])
    results_path = f"results/{problem}"
    os.makedirs(results_path, exist_ok=True)
    with open(f"{results_path}/monolithic_results.csv", "w") as f:
        results_df.to_csv(f, index=False)


def parse_stderr(stderr_decoded):
    n_transitions, max_states, winning_state_size, game_state_size = 0, 0, 0, 0
    if "*** ntransitions:" in stderr_decoded:
        n_transitions = int(stderr_decoded.split("*** ntransitions:")[1].split("\n")[0].strip())
    if "*** maxStates" in stderr_decoded:
        max_states = int(stderr_decoded.split("*** maxStates:")[1].split("\n")[0].strip())
    if "Winning state size:" in stderr_decoded:
        winning_state_size = int(stderr_decoded.split("Winning state size:")[1].split("\n")[0].strip())
    if "Game state size:" in stderr_decoded:
        game_state_size = int(stderr_decoded.split("Game state size:")[1].split("\n")[0].strip())
    return n_transitions, max_states, winning_state_size, game_state_size


def get_instances(base_input, problem):
    # Get all the instances in the lts folder for the given problem
    instances = []
    problem_path = f"{base_input}/{problem}"
    for file in os.listdir(problem_path):
        if file.endswith(".lts"):
            instances.append(file.replace(".lts", ""))
    return instances


def run_all(base_input):
    """
    Solves the base_input problem with monolithic and gets the controller and feature list results
    :return:
    """
    problems = [f"sokoban-{grid_size}-1" for grid_size in range(4, 8)]

    for problem in problems:
        # grab the instances from the lts folder, all lts inside the problem subfolder
        instances = get_instances(base_input, problem)
        run_monolithic_problem(base_input, problem, instances)


def remove_big_files():
    # Remove all files that are bigger than 100MB (for github)
    for root, dirs, files in os.walk("train_data/raw_data"):
        for file in files:
            if os.path.getsize(os.path.join(root, file)) > 100000000:
                print(f"Removing {os.path.join(root, file)}")
                os.remove(os.path.join(root, file))
                if "_features.csv" in file:
                    # also remove the controller file
                    controller_file = file.replace("_features.csv", "_controller.lts")
                    print(f"Removing {os.path.join(root, controller_file)}")
                    os.remove(os.path.join(root, controller_file))

if __name__ == '__main__':
    run_all(BASE_INPUT)
    # remove_big_files()
