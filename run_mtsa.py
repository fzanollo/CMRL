import argparse
import glob
import os
import platform
import re
import shutil
import subprocess

import pandas as pd

from utils.utils import run_problem_instance, discover_scenarios

def parse_stdout(stdout_decoded):
    # Define regex patterns for each identifier
    patterns = {
        'ExpandedStates': r'ExpandedStates:\s*(\d+)',
        'UsedStates': r'UsedStates:\s*(\d+)',
        'ExpandedTransitions': r'ExpandedTransitions:\s*(\d+)',
        'UsedTransitions': r'UsedTransitions:\s*(\d+)',
        'Elapsed in Synthesis': r'Elapsed in Synthesis:\s*(\d+)\s*ms',
        'findNewGoalsCalls': r'findNewGoalsCalls:\s*(\d+)',
        'findNewErrorsCalls': r'findNewErrorsCalls:\s*(\d+)',
        'propagateGoalsCalls': r'propagateGoalsCalls:\s*(\d+)',
        'propagateErrorsCalls': r'propagateErrorsCalls:\s*(\d+)',
        'maxMemoryUsed': r'maxMemoryUsed:\s*([\d.]+)\s*MB',
        'heuristicTime': r'heuristicTime:\s*(\d+)\s*ms',
        'Composition time': r'Composition time:\s*(\d+)\s*ms'
    }

    stats = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout_decoded)
        if match:
            stats[key] = match.group(1)
        else:
            stats[key] = None  # Or handle missing values as needed

    return stats


def run_instance(problem, instance_name, output_path, other_params, controller_name, lts_dir, timeout_seconds):
    # controller_name = MonolithicController for monolithic, DirectedController for DCS
    instance_filepath = f"{lts_dir}/{problem}/{instance_name}.lts"

    seed = None

    heuristic = "RA"
    if 'ml_model' in other_params:
        ml_model = other_params['ml_model']
        full_model_name = ml_model.replace(".onnx", "").split("/")[-1]
        # Extract just the meaningful part (remove problem prefix if present)
        # e.g., "sokoban-4-1_dqn_agent_2obstaclesC1" -> "dqn_agent_2obstaclesC1"
        heuristic = full_model_name.replace(f"{problem}_", "")
        extra_options = f"--ml_model {ml_model}"
    elif 'ml_model_same_origin' in other_params:
        heuristic = "two_models"
        ml_model_same_origin = other_params['ml_model_same_origin']
        ml_model_diff_origin = other_params['ml_model_diff_origin']
        extra_options = f"--ml_model_same_origin {ml_model_same_origin} --ml_model_diff_origin {ml_model_diff_origin}"
    elif 'seed' in other_params:
        heuristic = "random"
        seed = other_params['seed']
        extra_options = f"--seed {seed}"
    else:
        # run RA
        extra_options = ""

    output_filename = f"{instance_name}_{heuristic}_controller.fsp"
    execution_time, result, stdout_decoded, stderr_decoded = (
        run_problem_instance(instance_filepath, output_path, output_filename, controller_name,
                             extra_options=extra_options, timeout=timeout_seconds))

    if 'ONNXRuntimeError' in stderr_decoded or 'ORT_INVALID_ARGUMENT' in stderr_decoded:
        print(f"Error processing {instance_name}. ONNXRuntimeError")
        result = 'TO'
    elif 'ONNXRuntimeError' in stdout_decoded or 'ORT_INVALID_ARGUMENT' in stdout_decoded:
        print(f"Error processing {instance_name}. ONNXRuntimeError")
        result = 'TO'
    elif result == 'OK':
        match = re.search(r'Elapsed in Synthesis:\s+(\d+)\s+ms', stdout_decoded)
        if match:
            synthesis_time = match.group(1)
        else:
            print(f"Error processing {instance_name}. Synthesis time not found in stdout using regex!!")
            synthesis_time = None

        match = re.search(r'Using\s+(.*)', stderr_decoded)
        model_used = match.group(1).strip() if match else "not found"
        print(f"Using model: {model_used}")
        print(f"Problem {instance_name} correctly solved, synthesis time: {synthesis_time} ms")
    elif result == 'No controller':
        print(f"Problem {instance_name} correctly solved, no controller found")
    elif result == 'TO':
        print(f"Timeout processing {instance_name}")
    else:
        print(f"Error processing {instance_name}. Result: {result}")

    data = {
        'instance_name': instance_name,
        'execution_time': execution_time,
        'result': result
    }

    # add extra info from stdout
    data.update(parse_stdout(stdout_decoded))

    return data


def move_exploration_trace_files(controllers_outpath, results_outpath):
    results_outpath = os.path.join(results_outpath, 'exploration_trace')
    print(f"Moving exploration trace files from {controllers_outpath} to {results_outpath}")

    # Ensure the destination directory exists
    os.makedirs(results_outpath, exist_ok=True)

    # Use os.path.join for cross-platform glob pattern
    pattern = os.path.join(controllers_outpath, "*exploration_trace*")
    files = glob.glob(pattern)
    if not files:
        print("No exploration trace files found to move.")
        return

    if platform.system() == 'Windows':
        for file in files:
            dest_file = os.path.join(results_outpath, os.path.basename(file))
            if os.path.exists(dest_file):
                os.remove(dest_file)
            try:
                # Move to the explicit destination file path to avoid issues when the
                # destination directory didn't exist or other path quirks on Windows.
                shutil.move(file, dest_file)
            except Exception as e:
                print(f"Failed to move {file} to {dest_file}: {e}")
        print("Exploration trace files moved successfully (Windows).")
    else:
        # For Unix-like systems use find+mv as before but ensure proper quoting of paths
        command = f'find "{controllers_outpath}" -type f -name "*exploration_trace*" -exec mv -f {{}} "{results_outpath}"/ \\;'
        try:
            subprocess.run(command, shell=True, check=True)
            print("Exploration trace files moved successfully (Unix).")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")


def set_up_and_run_on_instance(problem, instance_name, tipo, model_name, lts_dir, timeout_seconds, other_params=None, is_dcs=True):
    if other_params is None:
        other_params = {}

    results_outpath = f"results/{problem}/{tipo}"
    os.makedirs(results_outpath, exist_ok=True)
    csv_path = f"{results_outpath}/results_{tipo}.csv"

    controllers_outpath = f"results/{problem}/controllers"
    os.makedirs(controllers_outpath, exist_ok=True)

    controller_name = "DirectedController" if is_dcs else "MonolithicController"

    result = run_instance(
        problem,
        instance_name,
        controllers_outpath,
        other_params,
        controller_name,
        lts_dir,
        timeout_seconds,
    )

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=result.keys())

    # Append the new result as a row using loc
    df.loc[len(df)] = [result.get(col, None) for col in df.columns]

    # Save the DataFrame back to CSV
    df.to_csv(csv_path, index=False)

    # move the exploration_trace to the results folder
    move_exploration_trace_files(controllers_outpath, results_outpath)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Run MTSA experiments with configurable input/model paths.")
    parser.add_argument(
        "--lts-dir",
        default=os.getenv("CMRL_LTS_DIR", "generated_sokoban_specs"),
        help="Directory containing per-problem .lts specs (default: generated_sokoban_specs or CMRL_LTS_DIR).",
    )
    parser.add_argument(
        "--trained-agents-dir",
        default=os.getenv("CMRL_TRAINED_AGENTS_DIR", "trained_agents"),
        help="Directory containing trained agent ONNX models (default: trained_agents or CMRL_TRAINED_AGENTS_DIR).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("CMRL_TIMEOUT_SECONDS", str(60 * 10))),
        help="Timeout in seconds per synthesis call (default: 600 or CMRL_TIMEOUT_SECONDS).",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    lts_dir = args.lts_dir
    trained_agents_dir = args.trained_agents_dir
    timeout_seconds = args.timeout_seconds

    print(
        f"Configuration: lts_dir={lts_dir}, trained_agents_dir={trained_agents_dir}, "
        f"timeout_seconds={timeout_seconds}"
    )

    problems = [f"sokoban-{grid_size}-1" for grid_size in range(4, 8)]

    for problem in problems:
        # Discover scenarios from the directory
        scenarios = discover_scenarios(lts_dir, problem)

        if not scenarios:
            print(f"No scenarios found for {problem}. Skipping.")
            continue
        
        print(f"Found {len(scenarios)} scenarios for {problem}: {scenarios}")
        scenarios_results_rows = []

        # remove old results file if exists
        for model in ['dqn_agent', 'prev_rl', 'RA']:
            results_path = f"results/{problem}/{model}/results_{model}.csv"
            if os.path.exists(results_path):
                os.remove(results_path)
        if os.path.exists(f"results/{problem}/comparative_results_all.csv"):
            os.remove(f"results/{problem}/comparative_results_all.csv")

        # run all scenarios for the problem
        for scenario in scenarios:
            instance_name = f"{problem}_{scenario}"

            # run new trained models:
            for episodes in [400, 1000]:
                model_name = f"{problem}_{episodes}e_dqn_agent_2obstaclesC1"
                other_params = {
                    'ml_model': f'{trained_agents_dir}/{problem}_{episodes}e_dqn_agent/{model_name}.onnx'
                }
                new_rl_result = set_up_and_run_on_instance(
                    problem,
                    instance_name,
                    f"dqn_agent_{episodes}e",
                    model_name,
                    lts_dir,
                    timeout_seconds,
                    other_params,
                )
                new_rl_result['Model'] = f'DQN Agent {episodes}e'
                new_rl_result['Scenario'] = scenario
                scenarios_results_rows.append(new_rl_result)

            # run trained model with previous RL approach (Tomi):
            other_params = {
               'prevRL': True,
               'ml_model': f'{trained_agents_dir}/prev_rl/sokoban-5-1_2obst.onnx'
            }
            prev_rl_result = set_up_and_run_on_instance(
                problem,
                instance_name,
                "prev_rl",
                "sokoban-5-1_2obst",
                lts_dir,
                timeout_seconds,
                other_params,
            )
            prev_rl_result['Model'] = 'Previous RL'
            prev_rl_result['Scenario'] = scenario
            scenarios_results_rows.append(prev_rl_result)

            # run RA:
            other_params = {}
            ra_result = set_up_and_run_on_instance(
                problem,
                instance_name,
                'RA',
                "RA",
                lts_dir,
                timeout_seconds,
                other_params,
            )
            ra_result['Model'] = 'RA'
            ra_result['Scenario'] = scenario
            scenarios_results_rows.append(ra_result)

        # Build a single DataFrame with a MultiIndex (Model, Scenario)
        combined_df = pd.DataFrame(scenarios_results_rows).set_index(['Model', 'Scenario'])
        combined_df.index.set_names(['Model', 'Scenario'], inplace=True)

        pd.set_option('display.max_columns', None)
        print(combined_df)

        os.makedirs(f"results/{problem}", exist_ok=True)
        combined_df.to_csv(f"results/{problem}/comparative_results_all.csv")
