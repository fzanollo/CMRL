import os

import pandas as pd

if __name__ == '__main__':
    problems = [f"sokoban-{grid_size}-1" for grid_size in range(4, 8)]

    for problem_name in problems:
        data_folder = f"train_data/{problem_name}"

        if not os.path.exists(data_folder):
            continue

        # load all different scenarios' information (features) and collect all uncontrollable and controllable actions
        scenario_files = [f for f in os.listdir(data_folder) if f.endswith("_features.csv")]

        all_uncontrollable_actions = set()
        all_controllable_actions = set()

        for scenario_file in scenario_files:
            data = pd.read_csv(f"train_data/{problem_name}/{scenario_file}")

            uncontrollable_actions = set(data.loc[data['is_controllable_action'] == False, 'action'].unique())
            controllable_actions = set(data.loc[data['is_controllable_action'] == True, 'action'].unique())

            all_uncontrollable_actions.update(uncontrollable_actions)
            all_controllable_actions.update(controllable_actions)

        # output to a csv file for each type of actions
        uncontrollable_actions_df = pd.DataFrame(sorted(list(all_uncontrollable_actions)), columns=['action'])
        controllable_actions_df = pd.DataFrame(sorted(list(all_controllable_actions)), columns=['action'])

        uncontrollable_actions_df.to_csv(f"train_data/{problem_name}/all_uncontrollable_actions.csv", index=False)
        controllable_actions_df.to_csv(f"train_data/{problem_name}/all_controllable_actions.csv", index=False)
