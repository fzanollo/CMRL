import random

from environment.environment import Environment
from utils.model_utils import load_model

if __name__ == '__main__':
    # Problem instance needs to be the same as in training
    problem_name = "sokoban-5-1"
    scenario = "2obstacles"
    instance_name = f"{problem_name}_{scenario}"

    model_name = f"{problem_name}_dqn_agent_0"
    model_path = f"trained_agents/{problem_name}_dqn_agent/{model_name}.pth"

    # Load environment
    env = Environment(problem_name, instance_name, {'interactive': True})

    # Load model
    agent = load_model(model_path, env)
    agent.epsilon = 0  # No exploration during evaluation

    # Set to evaluation mode after initialization
    agent.policy_net.eval()
    agent.target_net.eval()

    # Have the human play against the agent
    obs_state, info = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        print(f"Step {step}, Current State: {obs_state}")
        # get all the agent action rewards
        action_mask = info.get('action_mask', None)
        action_rewards = agent.action_rewards(obs_state, action_mask)
        agent_action_idx = max(action_rewards, key=action_rewards.get)
        agent_action = env.decode_action(agent_action_idx)

        # Agent step
        state_ranking, obs_successor, successor_ranking = env.transition_info_from_current_state(agent_action)
        print(f"\033[93m\tAgent choose transition: "
              f"{obs_state} (r:{state_ranking}) -- {agent_action} --> {obs_successor} (r:{successor_ranking})\033[0m")

        # Env step
        next_state, reward, terminated, truncated, info = env.step(env.encode_action(agent_action))
        done = terminated or truncated

        obs_state = next_state
        total_reward += reward
        step += 1

    is_win = info.get('is_win', False)
    if is_win:
        print(f"\033[92m\tAgent wins! Total reward: {total_reward}\033[0m")
    else:
        print(f"\033[91m\tAgent loses! Total reward: {total_reward}\033[0m")
