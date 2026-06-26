class CompStateInfo:
    def __init__(self, state_id, ranking, component_states, abstract_state):
        self.state_id = state_id
        self.ranking = ranking
        self.component_states = component_states
        self.abstract_state = abstract_state
        self.controllable_action_to_successor_id = {}
        self.uncontrollable_action_to_successor_id = {}
        self.available_controllable_actions_encoded = None

    def __str__(self):
        return (f"CompStateInfo(state_id={self.state_id}, ranking={self.ranking}, "
                f"component_states={self.component_states}, abstract_state={self.abstract_state}, "
                f"len controllable_actions = {len(self.controllable_action_to_successor_id)}, "
                f"len uncontrollable_actions = {len(self.uncontrollable_action_to_successor_id)})")

    def add_transition(self, transition, controllable):
        """Add a single transition to the state."""
        _, action, successor = transition
        if controllable:
            self.controllable_action_to_successor_id[action] = successor
        else:
            self.uncontrollable_action_to_successor_id[action] = successor

    def add_transitions_batch(self, transitions_df):
        """Add multiple transitions in batch for efficiency.
        :param transitions_df: DataFrame with multiindex (state, action, successor) and column 'is_controllable_action'
        """
        controllable_transitions = transitions_df[transitions_df['is_controllable_action'] == True]
        uncontrollable_transitions = transitions_df[transitions_df['is_controllable_action'] == False]

        for (state, action, successor), row in controllable_transitions.iterrows():
            self.controllable_action_to_successor_id[action] = successor
        for (state, action, successor), row in uncontrollable_transitions.iterrows():
            self.uncontrollable_action_to_successor_id[action] = successor

    def is_fully_uncontrollable(self):
        """Check if the state has only uncontrollable actions."""
        return len(self.controllable_action_to_successor_id) == 0 and len(self.uncontrollable_action_to_successor_id) > 0
