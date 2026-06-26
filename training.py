import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from DQNAgent import DQNAgent
from environment.environment import Environment
from utils.utils import plot_learning_curve, plot_winning_rate, save_results, plot_loss_curve
from utils.model_utils import save_agent


def train_dqn(env, episodes, agent_params, target_update_freq=10, verbose=False, agent: Optional[DQNAgent]=None):
    """
    If target_update_freq is set to None, the target network will be updated using soft updates.
    """
    start_time = time.time()
    if agent is None:
        print(f"Training DQN agent from scratch with parameters: {agent_params}")
        agent = DQNAgent(env, **agent_params)
    else:
        print("Continuing training of existing DQN agent.")
        # replace the agent's environment with the new one
        agent.replace_environment(env)

    episodes_info = {}
    episode_rewards = []

    for episode in range(episodes):
        print(f"\033[93mStarting episode {episode}/{episodes} ({env.problem_name})\033[0m")
        print(f"\tepsilon: {agent.epsilon:.2f}")
        episodes_info[episode] = {}
        obs_state, info = env.reset()
        total_reward = 0
        done = False
        step = 0
        episode_loss = 0.0

        episodes_info[episode]['starting_total_steps'] = agent.total_steps
        episodes_info[episode]['starting_epsilon'] = agent.epsilon

        while not done:  # Run one episode (steps until done)
            action_mask = info.get('action_mask', None)
            current_concrete_state = info['state_id']

            action = agent.select_action(obs_state, action_mask)
            obs_next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_action_mask = info.get('action_mask', None)
            next_concrete_state = info['state_id']
            agent.store_transition(obs_state, action, reward, obs_next_state, done, next_action_mask)
            train_result = agent.train()

            # Update the target network either by soft update or hard update every `target_update_freq` steps
            if target_update_freq is None:
                agent.soft_update_target_network()
            elif step % target_update_freq == 0:
                agent.update_target_network()

            if agent.epsilon_decay_steps is not None:
                agent.update_epsilon_by_steps()

            if train_result is not None:
                loss_val, mean_absolute_diff_Qs = train_result
                episode_loss += loss_val  # Accumulate loss

                if step % 100 == 0 or done:
                    state_ranking = env.get_ranking_from_id(current_concrete_state)
                    successor_ranking = env.get_ranking_from_id(next_concrete_state)
                    print(f"\033[93m\tGoing from ranking {state_ranking} to {successor_ranking}, reward: {reward:.2f}, "
                          f"with action '{env.decode_action(action)}'\033[0m")

                    print(f"\tEpisode {episode}, Step {step}, Loss: {loss_val:.4f}, "
                          f"MA Diff Qs: {mean_absolute_diff_Qs:.4f}, "
                          f"Epsilon: {agent.epsilon:.2f}, total steps: {agent.total_steps}, "
                          f"LR: {agent.optimizer.param_groups[0]['lr']:.6f}")

            obs_state = obs_next_state
            total_reward += reward
            step += 1

        is_win = info.get("is_win", False)
        episodes_info[episode]['is_win'] = is_win

        color = "\033[91m" if not is_win else "\033[92m"  # Red for loss, green for win
        reset = "\033[0m"
        result = "win" if is_win else "loss"
        print(f"{color}\tEpisode {episode} finished as {result} in {step+1} steps with reward: {total_reward}{reset}")

        episode_loss = episode_loss / max(1, step)  # Average loss per step
        episodes_info[episode]['avg_loss'] = episode_loss
        episodes_info[episode]['reward'] = total_reward
        episodes_info[episode]['win'] = is_win
        episodes_info[episode]['steps'] = step
        episodes_info[episode]['composition_size'] = (env.composition.number_of_nodes(),
                                                      env.composition.number_of_edges())

        episode_rewards.append(total_reward)
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}, "
                  f"Average Reward: {np.mean(episode_rewards[-100:]):.2f}, Epsilon: {agent.epsilon:.2f}")
            if env.composition.number_of_nodes() < 500:
                env.render()

        if agent.epsilon_decay_episodes is not None:
            agent.update_epsilon_by_episodes()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time / 60:.2f} minutes.")
    episodes_info_df = pd.DataFrame.from_dict(episodes_info, orient='index')
    return agent, episodes_info_df, training_time


def show_tau_info(env):
    total_states = len(env.state_id_to_info)
    multi_action_states = 0
    only_one_action_states = 0
    tau_only_states = 0
    for state_id, info in env.state_id_to_info.items():
        num_valid = len(info.controllable_action_to_successor_id)
        if num_valid > 1:
            multi_action_states += 1
        elif num_valid == 1:
            if list(info.controllable_action_to_successor_id.keys())[0] == "tau":
                tau_only_states += 1
            else:
                only_one_action_states += 1
    print(
        f"Total states: {total_states}, Multi-action: {multi_action_states} ({multi_action_states / total_states:.2%}), "
        f"Single-action: {only_one_action_states} ({only_one_action_states / total_states:.2%}), "
        f"Tau-only: {tau_only_states} ({tau_only_states / total_states:.2%})"
    )


def output_results(results_path, model_name, episodes_info, env):
    os.makedirs(results_path, exist_ok=True)
    # Plot the learning curve
    plot_learning_curve(model_name, results_path, episodes_info['reward'].tolist())
    # Plot the loss curve
    plot_loss_curve(model_name, results_path, episodes_info['avg_loss'].tolist())
    # Plot the winning rate
    plot_winning_rate(model_name, results_path, episodes_info['win'].tolist())
    save_results(episodes_info, env, results_path, model_name)


if __name__ == "__main__":
    dimensions = range(4, 7 +1)
    # dimensions = [7]
    problems = [f"sokoban-{grid_size}-1" for grid_size in dimensions]
    for episodes in [400, 1000]:  # Number of training episodes

        for problem_name in problems:
            print(f"\n\033[95m=== Training for problem: {problem_name} ===\033[0m\n")

            base_model_name = f"{problem_name}_{episodes}e_dqn_agent"

            # DQN agent parameters
            # target_update_freq = 100  # Frequency of hard update for the target network
            target_update_freq = None  # Use soft updates instead of hard updates

            agent_params = {
                "learning_rate": 0.0001,
                "gamma": 0.99,
                "epsilon": 1.0,
                "epsilon_min": 0.01,

                "epsilon_decay_episodes": episodes * 8//9,

                "buffer_size": 100000,
                "batch_size": 256,  # 128
                "tau": 0.001,
                "hidden_layers": [512, 256, 128],

                'use_priority_replay': True,
            }

            train_scenarios = ["2obstaclesC1"]

            agent = None
            train_from_scenario = 0

            for i in range(train_from_scenario, len(train_scenarios)):
                scenario  = train_scenarios[i]

                # TODO same starting epsilon or smaller when continuing training?

                print(f"\n\033[95m=== Training on scenario: {scenario} ===\033[0m\n")
                instance_name = f"{problem_name}_{scenario}"

                # Create the environment
                env_params = {
                    'verbose': False,
                    # 'verbose': True,
                    'uncontrollable_action_probability': 0.01,
                    'ignore_agent_action_probability': 0}

                env = Environment(problem_name, instance_name, args=env_params)

                model_name = f"{base_model_name}_{scenario}"
                results_path = f"trained_agents/{base_model_name}/training_results/"
                os.makedirs(results_path, exist_ok=True)

                model_path = f"trained_agents/{base_model_name}/{model_name}"

                agent, episodes_info, training_time = train_dqn(env, episodes, agent_params,
                                                                target_update_freq=target_update_freq, verbose=False, agent=agent)

                average_reward = episodes_info['reward'].mean()
                average_steps_per_episode = episodes_info['steps'].mean()
                # output training info to a csv (training_time, average_reward, average_steps)
                info = {
                    'training_time_seconds': training_time,
                    'average_reward': average_reward,
                    'average_steps_per_episode': average_steps_per_episode
                }
                info_df = pd.DataFrame([info])
                info_df.to_csv(f"{results_path}/{model_name}_training_info.csv", index=False)

                print(f"Training results for model '{model_name}':")
                print(f"\tTotal episodes: {len(episodes_info)}, Average reward: {average_reward:.2f}")
                print(f"\tAverage steps per episode: {average_steps_per_episode:.2f}")
                print(f"\tComposition size: {env.composition.number_of_nodes()}")

                # Save results and agent
                output_results(results_path, model_name, episodes_info, env)

                agent_params['target_update_freq'] = target_update_freq
                agent_params['episodes'] = episodes
                save_agent(agent, model_path, agent_params, instance_name, episodes, env)


