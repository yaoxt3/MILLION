from rlpyt.agents.dqn.dqn_agent import DqnAgent

class DqnAgentEnvSpaces(DqnAgent):

    def make_env_to_model_kwargs(self, env_spaces):
        """Generate any keyword args to the model which depend on environment interfaces."""
        return {'env_spaces': env_spaces}

    def initialize(self, *args, **kwargs):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None # don't let base agent try to initialize model
        super().initialize(*args, **kwargs)
        if _initial_model_state_dict is not None:
            self.model.load_state_dict(_initial_model_state_dict['model'])
            self.target_model.load_state_dict(_initial_model_state_dict['model'])
