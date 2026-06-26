import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.environment import Environment


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(DQN, self).__init__()
        self.device = kwargs.get('device', torch.device("cpu"))
        hidden_layers = kwargs.get('hidden_layers', [128, 64])
        in_dim = input_dim

        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x, hidden=None):
        # Accept 1D (single state) or 2D (batch, features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'action_mask', 'priority'))


class DQNAgent:
    def __init__(self, env: Environment, **kwargs):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.gamma = kwargs.get('gamma', 0.99)

        self.epsilon_start = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.epsilon_decay_steps = kwargs.get('epsilon_decay_steps', None)
        self.epsilon_decay_episodes = kwargs.get('epsilon_decay_episodes', None)
        # Only leave one epsilon update method
        assert not (self.epsilon_decay_steps and self.epsilon_decay_episodes), "Choose only one epsilon decay method: by steps or by episodes."

        self.batch_size = kwargs.get('batch_size', 64)
        self.tau = kwargs.get('tau', 0.005)  # Soft update parameter

        self.buffer_size = kwargs.get('buffer_size', 10000)
        self.use_priority_replay = kwargs.get('use_priority_replay', True)

        # Initialize networks
        kwargs['device'] = self.device
        self.input_dim = env.get_observation_space_size()
        self.action_space_size = env.action_space.n
        self.policy_net = DQN(self.input_dim, env.action_space.n, **kwargs).to(self.device)
        self.target_net = DQN(self.input_dim, env.action_space.n, **kwargs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer and learning rate scheduler
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.use_learning_rate_scheduler = kwargs.get('use_learning_rate_scheduler', False)
        if self.use_learning_rate_scheduler:
            print("Using learning rate scheduler.")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=kwargs.get('scheduler_factor', 0.7),
                patience=kwargs.get('scheduler_patience', 100),
                threshold=kwargs.get('scheduler_threshold', 0.01),
                cooldown=kwargs.get('scheduler_cooldown', 10),
                min_lr=kwargs.get('scheduler_min_lr', 1e-6)
            )

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss, more robust to outliers

        self.q_history = deque(maxlen=2)

        self.priority_epsilon = kwargs.get('priority_epsilon', 0.01)
        self.max_num_actions = kwargs.get('max_num_actions', None)
        self.reset_exploration_params()

    def reset_exploration_params(self):
        self.epsilon = self.epsilon_start
        self.total_steps = 0  # Global step counter
        self.episodes_counter = 0  # Episode counter
        if self.use_priority_replay:
            # Priority experience replay setup (PER)
            print("Using prioritized experience replay.")
            # Use deque for O(1) appends/pops and fixed maxlen
            self.memory = deque(maxlen=self.buffer_size)      # store Transition
            self.priorities = deque(maxlen=self.buffer_size)  # parallel priorities
            if self.max_num_actions is None:
                self.max_num_actions = len(self.env.controllable_actions)
        else:
            self.memory = deque(maxlen=self.buffer_size)
            self.priorities = None

    def replace_environment(self, new_env: Environment):
        # check, new input size and action space size should match
        assert new_env.get_observation_space_size() == self.input_dim, "New environment observation space size does not match."
        assert new_env.action_space.n == self.action_space_size, "New environment action space size does not match."
        self.env = new_env
        # TODO is ok to also reset the exploration params?
        # maybe is a nice idea to try both and compare
        self.reset_exploration_params()

    def get_q_values(self, states):
        # create tensor directly on device to avoid extra host->device copies
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return self.policy_net(states_t)

    def update_q_history(self, states):
        q_values = self.get_q_values(states).detach().cpu().numpy()
        if len(self.q_history) == 2:
            self.q_history.popleft()
        self.q_history.append(q_values)

    def mean_absolute_diff_between_Qs(self):
        if len(self.q_history) < 2:
            return False
        q_prev, q_curr = self.q_history[0], self.q_history[1]
        diff = np.abs(q_curr - q_prev).mean()  # Mean absolute difference
        return diff

    def select_action(self, state, action_mask):
        if action_mask is None:
            action_mask = np.ones(self.env.action_space.n, dtype=np.float32)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if random.random() < self.epsilon:
            # Explore "following the solution" using the inverse ranking as probabilities
            valid_indices = np.where(np.array(action_mask) == 1)[0]
            if len(valid_indices) == 0:
                return random.randint(0, self.env.action_space.n - 1)
            elif len(valid_indices) == 1:
                return valid_indices[0]
            actions_with_rankings = self.env.get_current_state_transitions_rankings()
            weights = []
            for idx in valid_indices:
                action = self.env.decode_action(idx)
                ranking = actions_with_rankings.get(action)  # All valid actions should be present
                weights.append(1 / (ranking + 1))  # Inverse ranking
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize to probabilities
            action = np.random.choice(valid_indices, p=weights)
            return int(action)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                masked_q_values = q_values + (action_mask_t - 1) * 1e9
                action = torch.argmax(masked_q_values, dim=1).item()
            return int(action)

    def action_rewards(self, state, action_mask):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            # Apply mask: set invalid actions to a large negative value
            masked_q_values = q_values + (action_mask_t - 1) * 1e9
        q_values_np = masked_q_values.cpu().numpy().squeeze(0)
        valid = np.where(np.array(action_mask) == 1)[0]
        res = dict(zip(valid.tolist(), q_values_np[valid].tolist()))
        return res

    def store_transition(self, state, action, reward, next_state, done, action_mask):
        # Compute a simple priority and append to parallel deques
        priority = 1.0
        if self.use_priority_replay:
            num_actions = np.sum(action_mask)
            priority = (num_actions / self.max_num_actions) + self.priority_epsilon
        transition = Transition(np.array(state), action, reward, np.array(next_state), done, np.array(action_mask), priority)
        self.memory.append(transition)
        if self.use_priority_replay:
            self.priorities.append(priority)

    def update_epsilon_by_steps(self):
        self.total_steps += 1
        if self.epsilon_decay_steps is not None:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (
                                       self.total_steps / self.epsilon_decay_steps))
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_epsilon_by_episodes(self):
        self.episodes_counter += 1
        if self.epsilon_decay_episodes is not None:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (
                                       self.episodes_counter / self.epsilon_decay_episodes))
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        if self.use_priority_replay:
            # build probabilities once from the priorities deque
            priorities = np.array(self.priorities, dtype=np.float64)
            priorities = np.nan_to_num(priorities, nan=0.0, posinf=0.0, neginf=0.0)
            priorities = np.clip(priorities, 0.0, None)

            total = priorities.sum()
            if total <= 0 or not np.isfinite(total):
                probs = np.ones(len(self.memory), dtype=np.float64) / len(self.memory)
            else:
                probs = priorities / total

            # Final safe renormalize to avoid tiny floating errors
            probs = probs / probs.sum()

            indices = np.random.choice(len(self.memory), self.batch_size, p=probs, replace=False)
            batch = [self.memory[i] for i in indices]
        else:
            batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones, next_action_masks, _ = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device)
        next_action_masks = torch.as_tensor(np.array(next_action_masks), dtype=torch.float32, device=self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_action_values = self.policy_net(next_states)
            masked_next_action_values = next_action_values + (next_action_masks - 1) * 1e9
            next_actions = masked_next_action_values.argmax(1).unsqueeze(1)
            next_target_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            expected_q_values = rewards + (1 - dones) * self.gamma * next_target_q_values

        loss = self.loss_fn(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_learning_rate_scheduler:
            self.scheduler.step(loss.item())
        return loss.item(), self.mean_absolute_diff_between_Qs()

    # Only use one of the following two methods:
    def soft_update_target_network(self):
        # Efficient in-place soft update without state_dict copies
        with torch.no_grad():
            for p_policy, p_target in zip(self.policy_net.parameters(), self.target_net.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(p_policy.data * self.tau)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
