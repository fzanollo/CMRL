from abc import ABC, abstractmethod
from typing import List, Sequence, Optional

import gymnasium as gym
import numpy as np


class ObservationSpaceBase(ABC):
    """Abstract observation-space handler API."""
    def __init__(self, env):
        self.state_id_to_info = env.state_id_to_info
        self.observation_space: Optional[gym.Space] = None
        self.build_space()

    @abstractmethod
    def build_space(self) -> None:
        """Create and assign self.observation_space."""
        raise NotImplementedError

    @abstractmethod
    def encode_state(self, state_id: int) -> np.ndarray:
        """Return observation encoding for given concrete state id."""
        raise NotImplementedError

    @abstractmethod
    def decode_state(self, obs: Sequence[int]) -> List[int]:
        """Return list of state ids matching the observation encoding."""
        raise NotImplementedError

    @abstractmethod
    def get_size(self) -> int:
        """Return size of the observation space."""
        raise NotImplementedError

    def get_space(self) -> gym.Space:
        if self.observation_space is None:
            self.build_space()
        return self.observation_space


# python
class ComponentObservationSpace(ObservationSpaceBase):
    """
    Observation is the raw component_states vector (e.g. [0,1,6,1,0]).
    Now returns a normalized float vector in encode_state (values in [0,1]).
    """
    def __init__(self, env):
        self.component_number_of_states = env.component_number_of_states
        # normalization factors: max index per component (n_states - 1), avoid div by zero
        self.norm_factors = np.array([max(1, n - 1) for n in self.component_number_of_states], dtype=np.float32)
        super().__init__(env)

    def build_space(self) -> None:
        # keep original discrete space for compatibility, but agent will receive normalized floats
        self.observation_space = gym.spaces.MultiDiscrete(self.component_number_of_states)

    def encode_state(self, state_id: int) -> np.ndarray:
        comp_states = self.state_id_to_info[state_id].component_states
        arr = np.array(comp_states, dtype=np.float32)
        return arr / self.norm_factors

    def decode_state(self, obs: Sequence[int]) -> List[int]:
        return list(obs)

    def get_size(self) -> int:
        return int(self.observation_space.nvec.size)


class SubstatesAbstractionObservationSpace(ObservationSpaceBase):
    """
    Abstract (0, 1, +1) machines in a state for each 'component type' and state number
    where component type is grouped by the name + max state number
    so, given a state we get (0, 1, +1) for each possible state of each component type
    state: [0, 1, 2, 0, 0] where components are A, B, C, B, A
    component types: (A, 1), (B, 2), (C, 3), so A: (0, +1), B: (0, 1), (1, 1), C: (0, 0), (1, 0), (2, 1)
    observation state: [+1, 1, 1, 0, 0, 1]
    """
    def __init__(self, env):
        self.components_names = env.components_names
        self.component_number_of_states = env.component_number_of_states
        assert len(self.components_names) == len(self.component_number_of_states)

        self.encoding_to_state = None
        self.state_to_encoding = None
        super().__init__(env)

    def build_space(self) -> None:
        # group component indices by (component_name, n_states) with a stable order
        component_groups = {}
        for idx, (name, n_states) in enumerate(zip(self.components_names, self.component_number_of_states)):
            key = (name, n_states)
            component_groups.setdefault(key, []).append(idx)
        # keep a deterministic order for flattening
        component_groups = sorted(component_groups.items())

        # build forward encoding: for each concrete state, count how many components
        # are in each substate for each component group, clamped to 2
        self.state_to_encoding = {}
        for state_id, info in self.state_id_to_info.items():
            flattened = []
            for (name, n_states), indices in component_groups:
                counts = [0] * n_states
                for i in indices:
                    s = info.component_states[i]
                    counts[s] = min(counts[s] + 1, 2)
                flattened.extend(counts)
            self.state_to_encoding[state_id] = np.array(flattened, dtype=np.int32)

        self.observation_space = gym.spaces.MultiDiscrete([3]*len(self.state_to_encoding[0]))

        # build reverse map from encoding tuples to concrete state ids
        self.encoding_to_state = {}
        for state_id, encoding in self.state_to_encoding.items():
            key = tuple(int(x) for x in encoding)  # explicit conversion to an immutable key
            self.encoding_to_state.setdefault(key, []).append(state_id)

        # print proportion of collisions
        collisions = sum(1 for v in self.encoding_to_state.values() if len(v) > 1)
        print(f"Percentage of collisions in state encoding: {collisions / len(self.state_id_to_info):.2%} ")

    def encode_state(self, state_id: int) -> np.ndarray:
        return self.state_to_encoding[state_id]

    def decode_state(self, obs: Sequence[int]) -> List[int]:
        states_ids = self.encoding_to_state[tuple(int(x) for x in obs)]
        return states_ids

    def get_size(self) -> int:
        return int(self.observation_space.nvec.size)

