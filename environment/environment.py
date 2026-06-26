import random

import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from environment.observation_spaces import ComponentObservationSpace, ObservationSpaceBase, \
    SubstatesAbstractionObservationSpace
from utils.comp_state_info import CompStateInfo
from utils.utils import get_instance_data


class Environment(gym.Env):
    def __init__(self, problem_name, instance_name, args=None):
        """Initialize the environment with the given problem name, n, and k."""
        super().__init__()
        self.problem_name = problem_name
        self.instance_name = instance_name

        if args is None:
            args = {}

        self.uncontrollable_action_probability = args.get('uncontrollable_action_probability', 0.5)
        self.ignore_agent_action_probability = args.get('ignore_agent_action_probability', 0)

        self.verbose = args.get('verbose', False)
        self.interactive = args.get('interactive', False)

        (state_id_to_info, transitions_df, uncontrollable_actions, controllable_actions,
         all_roles, component_number_of_states) = get_instance_data(problem_name, instance_name)

        self.state_id_to_info: dict[int, CompStateInfo] = state_id_to_info
        self.transitions_df = transitions_df
        self.uncontrollable_actions = list(sorted(uncontrollable_actions))
        self.controllable_actions = list(sorted(controllable_actions))
        self.component_number_of_states = component_number_of_states

        self.rankings_values = sorted(set(info.ranking for info in self.state_id_to_info.values()))
        self.infinite_ranking = self.rankings_values[-1]
        self.worst_ranking = self.rankings_values[-2]
        self.norm_infinity_ranking = self.worst_ranking + 1 # worst ranking before infinity + 1
        # change all infinite rankings to worst_ranking
        # this way the difference to a loosing state is in the same range as other differences,
        # effectively normalizing it. Ex. DP 3-1 has infinite ranking = 1050, worst ranking before that is 27
        for info in self.state_id_to_info.values():
            if info.ranking == self.infinite_ranking:
                info.ranking = self.norm_infinity_ranking

        # assert there's no remaining "infinite ranking"
        for info in self.state_id_to_info.values():
            assert info.ranking != self.infinite_ranking

        self.best_reward = self.worst_ranking + 1

        self.components_names = [abs_st.split('_')[0] for abs_st in state_id_to_info.get(0).abstract_state]
        print("Component number of states:", component_number_of_states)
        print(f"\twith component names: {self.components_names}")

        # Action space -----------------------------------------------------
        self.all_actions = self.controllable_actions + self.uncontrollable_actions
        # Encode all actions but...
        self.action_to_encoding = {action: idx for idx, action in enumerate(self.all_actions)}
        # ...only controllable actions for the agent
        self.action_space = gym.spaces.Discrete(len(self.controllable_actions))

        self.uncontrollable_visited_count = {}

        for state_info in self.state_id_to_info.values():
            # Encode agent available actions in each state for quick access
            state_info.available_controllable_actions_encoded = [self.action_to_encoding[action] for action
                                                                 in state_info.controllable_action_to_successor_id.keys()]
            # Uncontrollable visit counter for weighted random when choosing uncontrollable actions
            uncontrollable_actions = state_info.uncontrollable_action_to_successor_id.keys()
            self.uncontrollable_visited_count[state_info.state_id] = {action: 0 for action in uncontrollable_actions}

        # Observation space ------------------------------------------------
        self.observation_space: ObservationSpaceBase = None

        # Option 1: Observation is the raw component_states vector
        self.observation_space = ComponentObservationSpace(self)

        # option 2: Abstract observation space grouping by component "types" and number of states
        # self.observation_space = SubstatesAbstractionObservationSpace(self)

        # State space ------------------------------------------------------
        #  Concrete states of the plant
        self.state_space = gym.spaces.Discrete(len(self.state_id_to_info))

        self.explored_transitions = set()

        # Reward max steps ----------------------------------------------
        # we are giving the agent +50% of worst ranking "missed" steps to get to the goal from a "worst ranking state"
        self.max_steps = self.worst_ranking + int(0.5 * self.worst_ranking)

        # Composition variables
        self.current_state = 0
        self.composition = nx.DiGraph()
        self.step_count = 0
        self.reset()

    def log(self, msg):
        if self.verbose:
            print(msg)

    def get_observation_space_size(self):
        return self.observation_space.get_size()

    def encode_state(self, state_id):
        return self.observation_space.encode_state(state_id)

    def decode_state(self, obs_state):
        return self.observation_space.decode_state(obs_state)

    def reset(self, **kwargs):
        """Reset the environment to an initial state."""
        self.log(f"Environment reset.") # TODO add episode info to logger
        self.step_count = 0
        self.current_state = 0
        self.composition = nx.DiGraph()
        action_mask = self.get_valid_actions_mask_from_id(self.current_state)
        info = {'action_mask': action_mask, 'state_id': self.current_state}
        return self.encode_state(self.current_state), info

    def get_all_controllable_actions_encoded(self):
        """Get all controllable actions encoded."""
        return [self.action_to_encoding[action] for action in self.controllable_actions]

    def get_valid_actions_from_id(self, state_id):
        """Get all valid controllable actions for the given state."""
        state_info = self.state_id_to_info[state_id]
        return list(state_info.controllable_action_to_successor_id.keys())

    def get_valid_actions_mask_from_id(self, state_id):
        action_mask = np.zeros(self.action_space.n, dtype=np.float32)
        action_mask[self.get_valid_actions_encoded_from_id(state_id)] = 1
        # assert not all zero
        assert np.any(action_mask), f"State {state_id} has no valid actions, action mask is all zeros."
        return action_mask

    def get_valid_actions_for_current_state(self):
        """Get all valid controllable actions for the current state."""
        return self.get_valid_actions_from_id(self.current_state)

    def get_valid_actions_encoded_from_id(self, state_id):
        """Get all valid controllable actions encoded for the given state."""
        state_info = self.state_id_to_info[state_id]
        return state_info.available_controllable_actions_encoded

    def encode_action(self, action):
        if action in self.action_to_encoding:
            return self.action_to_encoding[action]
        else:
            raise ValueError(f"Action {action} is not a valid action.")

    def decode_action(self, action_idx):
        if 0 <= action_idx < len(self.all_actions):
            return self.all_actions[action_idx]
        else:
            raise ValueError(f"Action index {action_idx} is not a valid action index.")

    def find_successor(self, state_id, action):
        state_info = self.state_id_to_info[state_id]
        return state_info.controllable_action_to_successor_id[action]

    def get_current_state_transitions_rankings(self):
        """Get the rankings of the successors for each valid action from the current state."""
        state_info = self.state_id_to_info[self.current_state]
        action_rankings = {}
        for action, successor_id in state_info.controllable_action_to_successor_id.items():
            successor_ranking = self.state_id_to_info[successor_id].ranking
            action_rankings[action] = successor_ranking
        return action_rankings

    def step(self, action_idx):
        """Apply an action to the environment and return the next state, reward, done flag,
        and additional info."""
        self.step_count += 1

        # Given the action index, retrieve the action
        action = self.decode_action(action_idx)
        state_id = self.current_state

        immediate_successor_id, after_env_successor_id = self._resolve_action(state_id, action)
        # immediate_successor_id = state from where the agent took the action
        # after_env_successor_id = state after the action and (maybe) uncontrollable action(s)

        self.explored_transitions.add((state_id, action, immediate_successor_id))

        # Calculate the reward
        reward, terminated, is_win, truncated = self.get_reward_and_terminated(state_id, action, immediate_successor_id,
                                                                    after_env_successor_id)

        self.log(f"Agent: state {state_id}, action '{action}', reward {reward:.3f}, terminated {terminated}")

        # If we reached a losing state, stay in the previous state, the episode does not finish
        if self._is_losing_state(after_env_successor_id) or self._is_losing_state(immediate_successor_id):
            next_state = state_id
        else:
            next_state = after_env_successor_id

        # Update the env
        self.current_state = next_state

        # Create action mask for next state
        action_mask = self.get_valid_actions_mask_from_id(self.current_state)

        info = {'action_mask': action_mask,
                'state_id': self.current_state,
                'previous_state_info': self.state_id_to_info[state_id],
                'next_state_info': self.state_id_to_info[self.current_state],
                'composition': self.composition,
                'is_win': is_win
                }
        # TODO can we add information about the "next_state"'s successors here?
        #  like (action, successor_state, ranking)? to be used by the agent?

        return self.encode_state(self.current_state), reward, terminated, truncated, info

    def _resolve_action(self, state_id, action):
        # Is a valid action from this state?
        current_state_info = self.state_id_to_info[self.current_state]

        if action not in current_state_info.controllable_action_to_successor_id:
            # Just skip to choosing uncontrollable actions
            final_succ_id = self.environment_turn_actions(state_id, True)
            successor_id = final_succ_id
            # self.log(f"{state_id} - {action} -> {final_succ_id} [invalid action, skipped]")
        else:
            # There's a possibility that the env does not play the agent action
            # TODO unless the state is a fully controllable state?
            if random.random() < self.ignore_agent_action_probability:
                # self.log(f"{state_id} - {action} -> {state_id} [action ignored]")
                successor_id = state_id
            else:
                successor_id = self.find_successor(self.current_state, action)
                self.log(f"{state_id} - {action} -> {successor_id} [controllable]")
                # self.update_composition_graph(state_id, successor_id, action)

            # After the action, we do a random choice for uncontrollable actions from the successor state
            final_succ_id = self.environment_turn_actions(successor_id)
        return successor_id, final_succ_id

    def is_deadlock(self, state_id):
        """Check if the given state is a deadlock."""
        state_info = self.state_id_to_info[state_id]
        return len(state_info.controllable_action_to_successor_id) == 0 and \
               len(state_info.uncontrollable_action_to_successor_id) == 0

    def render(self, mode='human'):
        """Render the composition so far.
        With the start state in orange, current state in blue,
        deadlock in red, and goal states in green."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.composition)  # TODO better layout?

        # Draw nodes with different colors based on their state
        color_map = {
            'start': 'orange',
            'current': 'blue',
            'deadlock': 'red',
            'goal': 'green',
            'other': 'lightgray'
        }
        node_colors = []
        for node in self.composition.nodes:
            if node == 0:
                node_colors.append(color_map['start'])
            elif self.is_deadlock(node):
                node_colors.append(color_map['deadlock'])
            elif self.state_id_to_info[node].ranking == 0:
                node_colors.append(color_map['goal'])
            elif node == self.current_state:
                node_colors.append(color_map['current'])
            else:
                node_colors.append(color_map['other'])

        # Add a legend with the colors
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map['start'], markersize=10, label='Start State'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map['current'], markersize=10, label='Current State'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map['deadlock'], markersize=10, label='Deadlock State'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map['goal'], markersize=10, label='Goal State'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color_map['other'], markersize=10, label='Other States')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        nx.draw(self.composition, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10)
        nx.draw_networkx_edge_labels(self.composition, pos,
                                     edge_labels=nx.get_edge_attributes(self.composition, 'action'))

        plt.title(f"Composition Graph - {self.instance_name}")
        plt.show()

    def close(self):
        """Close the environment."""
        pass

    def update_composition_graph(self, state, successor, action):
        self.composition.add_node(state)
        self.composition.add_node(successor)
        self.composition.add_edge(state, successor, action=action)

    def environment_turn_actions(self, state_id, force_one_action=False):
        if self.interactive:
            return self.human_input_env_action(state_id)

        env_trace_log = []
        successor_id = state_id  # Start with the current state
        stop = False
        while not stop:
            # Get the state info
            state_info = self.state_id_to_info[successor_id]

            # if we are in a goal or losing state, stop
            if self._is_goal_state(successor_id) or self._is_losing_state(successor_id):
                return successor_id

            # If there are uncontrollable actions, choose one with probability uncontrollable_action_probability
            if len(state_info.uncontrollable_action_to_successor_id) > 0:
                # If we are forcing one uncontrollable action, or the random choice is successful
                if random.random() < self.uncontrollable_action_probability or force_one_action:
                    force_one_action = False
                    successor_id, uncontrollable_action = self._play_uncontrollable_action(successor_id)
                    env_trace_log.append((uncontrollable_action, successor_id))
                    stop = True  # Stop after one uncontrollable action
                else:
                    # No uncontrollable action taken
                    stop = True
            else:
                # No uncontrollable actions to choose from
                # print(f"No uncontrollable actions available from state {successor_id}.")
                stop = True

            # Even if we should stop, continue if the successor is a state that only has 'Tau' as controllable action
            # Unless it's a goal or deadlock state
            # play on mixed states (env always wins the race)
            if self.is_mixed_state(successor_id):
                if not (self._is_goal_state(successor_id) or
                        self._is_losing_state(successor_id)):
                    stop = False
                    force_one_action = True

        self.log(f"Environment actions trace: {env_trace_log}, final state: {successor_id}")
        return successor_id

    def _play_uncontrollable_action(self, state_id):
        state_info = self.state_id_to_info[state_id]
        uncontrollable_actions = list(state_info.uncontrollable_action_to_successor_id.keys())
        visited_counts = self.uncontrollable_visited_count[state_id]

        # Choose an uncontrollable action randomly, weighted by the number of times each action has been taken
        # Probability is higher for actions with fewer visits
        weights = [1 / (visited_counts[action] + 1) for action in uncontrollable_actions]
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]
        uncontrollable_action = random.choices(uncontrollable_actions, weights=weights, k=1)[0]

        visited_counts[uncontrollable_action] += 1

        previous = state_id
        successor_id = state_info.uncontrollable_action_to_successor_id[uncontrollable_action]

        self.log(f"{previous} - {uncontrollable_action} -> {successor_id} [uncontrollable]")
        self.update_composition_graph(previous, successor_id, uncontrollable_action)
        return successor_id, uncontrollable_action

    def human_input_env_action(self, state_id):
        state_info = self.state_id_to_info[state_id]
        print(f"\tCurrent state: {self.encode_state(state_id)} "
              f"(concrete: {state_info.component_states}, r:{state_info.ranking})")

        # Show the set of controllable actions for that state (action names only)
        print("\tAvailable controllable actions:")
        for action in state_info.controllable_action_to_successor_id.keys():
            successor_id = state_info.controllable_action_to_successor_id[action]
            successor_info = self.state_id_to_info[successor_id]
            print(f"\t\t{action} -> {successor_info.component_states} (r:{successor_info.ranking})")

        # Show the env action options
        enumerated_actions = {idx: action for idx, action in
                              enumerate(state_info.uncontrollable_action_to_successor_id.keys())}

        if len(enumerated_actions) == 0:
            print("\tNo uncontrollable actions available from this state. Skipping.")
            return state_id
        else:
            print("\tAvailable uncontrollable actions:")
            for idx, action in enumerated_actions.items():
                successor_id = state_info.uncontrollable_action_to_successor_id[action]
                successor_info = self.state_id_to_info[successor_id]
                print(f"\t\t{idx}: {action} -> {successor_info.component_states} (r:{successor_info.ranking})")

            # Ask the user for input until a valid option is provided
            while True:
                choice = input("\tChoose an uncontrollable action by its number (or press Enter to skip): ")
                if choice == "":
                    print("\tNo uncontrollable action taken.")
                    return state_id
                if choice.isdigit() and int(choice) in enumerated_actions:
                    action = enumerated_actions[int(choice)]
                    successor_id = state_info.uncontrollable_action_to_successor_id[action]
                    successor_info = self.state_id_to_info[successor_id]
                    self.update_composition_graph(state_id, successor_id, action)
                    print(f"\033[93m\tUncontrollable action taken: {action} -> {successor_info.component_states}\033[0m")
                    return successor_id
                print("\tInvalid choice. Please try again.")

    def get_reward_and_terminated(self, state_id, action, immediate_successor_id, after_env_successor_id):
        """Get the reward and termination status for the given state, action, and successor."""
        state_info = self.state_id_to_info[state_id]

        # Our definition:
        immediate_reward = self.reward_diff_rankings(state_id, immediate_successor_id)
        after_env_reward = self.reward_diff_rankings(state_id, after_env_successor_id)

        imm_done, imm_is_win = self.is_done_and_win(immediate_successor_id)
        done, is_win = self.is_done_and_win(after_env_successor_id)

        # Reward = (alpha * immediate reward) + ((1 - alpha) * after env reward)
        alpha = 0.7  #  TODO tune these value
        reward = alpha * immediate_reward + (1 - alpha) * after_env_reward

        # small negative reward to encourage shorter paths
        reward -= 0.01

        # minigrid def
        # reward = 0
        # done = False
        # is_win = False
        # if self._is_losing_state(after_env_successor_id):
        #     reward = -0.001
        # elif self._is_goal_state(after_env_successor_id):
        #     done = True
        #     is_win = True
        #     reward = 1 - 0.9 * (self.step_count / self.max_steps)
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True

        return reward, done, is_win, truncated

    def reward_diff_rankings(self, state_id, successor_id):
        state_ranking = self.state_id_to_info[state_id].ranking
        successor_ranking = self.state_id_to_info[successor_id].ranking
        # the reward is the difference in ranking between the successor and the current state
        reward = state_ranking - successor_ranking

        if self._is_losing_state(successor_id):
            reward = -self.best_reward
        elif self._is_goal_state(successor_id):
            reward = self.best_reward
        return reward

    def is_done_and_win(self, state_id):
        """Check if the state is a terminal state (goal or losing)."""
        done = False
        is_win = False
        if self._is_goal_state(state_id):
            done = True
            is_win = True
        # elif self._is_losing_state(state_id):
        #     done = True
        return done, is_win

    def only_tau_action(self, state_id):
        """Check if the state has only one controllable action, which is 'Tau'."""
        state_info = self.state_id_to_info[state_id]
        return len(state_info.controllable_action_to_successor_id) == 1

    def is_mixed_state(self, state_id):
        # a state is mixed if it has uncontrollable and controllable actions
        # checking only if it has uncontrollable actions is enough
        state_info = self.state_id_to_info[state_id]
        return len(state_info.uncontrollable_action_to_successor_id) > 0

    def fully_uncontrollable_state(self, state_id):
        # a state is fully uncontrollable if it has uncontrollable actions and no controllable actions (or only 'Tau')
        state_info = self.state_id_to_info[state_id]
        return (len(state_info.uncontrollable_action_to_successor_id) > 0 and
                (len(state_info.controllable_action_to_successor_id) == 0 or
                 (len(state_info.controllable_action_to_successor_id) == 1 and
                  'Tau' in state_info.controllable_action_to_successor_id)))

    def _is_losing_state(self, state_id):
        """Check if the state is a losing state (deadlock or worst ranking)."""
        state_ranking = self.get_ranking_from_id(state_id)
        return self.is_deadlock(state_id) or state_ranking == self.norm_infinity_ranking

    def _is_goal_state(self, state_id):
        """Check if the state is a goal state (ranking 0)."""
        state_ranking = self.get_ranking_from_id(state_id)
        return state_ranking == 0

    def transition_info_from_current_state(self, agent_action):
        state_info = self.state_id_to_info[self.current_state]
        state_ranking = state_info.ranking

        if agent_action not in state_info.controllable_action_to_successor_id:
            raise ValueError(f"Action {agent_action} not valid in state {self.current_state}")

        next_state = state_info.controllable_action_to_successor_id[agent_action]
        successor_info = self.state_id_to_info[next_state]
        successor_ranking = successor_info.ranking

        obs_successor = self.encode_state(next_state)

        return state_ranking, obs_successor, successor_ranking

    def get_ranking_from_id(self, state_id):
        return self.state_id_to_info[state_id].ranking

