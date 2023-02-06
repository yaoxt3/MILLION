import numpy as np
import torch
from rlpyt.models.utils import update_state_dict
from rlpyt.agents.base import AgentStep
from learning_to_be_taught.recurrent_sac.sac_attention_model import PiAttentionModel, QAttentionModel
from rlpyt.utils.buffer import buffer_func, buffer_method
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.agents.base import RecurrentAgentMixin
from rlpyt.distributions.gaussian import DistInfoStd
from rlpyt.agents.base import AgentInputs
from learning_to_be_taught.recurrent_sac.transformer_model import PiTransformerModel, QTransformerModel
from learning_to_be_taught.models.transformer_models import DemonstrationTransformerModel
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.models.mlp import MlpModel
from rlpyt.models.qpg.mlp import QofMuMlpModel, PiMlpModel
from learning_to_be_taught.recurrent_sac.transformer_with_pi_head import TransformerWithPiHead


AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "prev_rnn_state"])
# AgentInputs = namedarraytuple("ModelInput", ["observation", "prev_action", 'prev_reward'])
RnnState = namedarraytuple('RnnState', ['pi', 'q1', 'q2'])
QValues = namedarraytuple('QValues', ['q1', 'q2', 'target_q1', 'target_q2'])
QRnnState = namedarraytuple('QRnnState', ['q1', 'q2', 'target_q1', 'target_q2'])

MIN_LOG_STD = -20
MAX_LOG_STD = 2


class EfficientRecurrentSacAgent(RecurrentAgentMixin, SacAgent):
    """Base agent for recurrent DQN (to add recurrent mixin)."""

    def __init__(self,
                 # ModelCls=DemonstrationTransformerModel,
                 # ModelCls=PiTransformerModel,
                 # ModelCls=MlpModel,
                 ModelCls=TransformerWithPiHead,
                 QModelCls=QTransformerModel,
                 shared_layer_size=512,
                 **kwargs):
        """
        Arguments are saved but no model initialization occurs.

        Args:
            ModelCls: The model class to be used.
            model_kwargs (optional): Any keyword arguments to pass when instantiating the model.
            initial_model_state_dict (optional): Initial model parameter values.
        """
        save__init__args(locals())

        self.model_kwargs = dict(num_features=self.shared_layer_size, size='medium', style='reordered')
        # self.model_kwargs = dict(hidden_sizes=[256, 256])
        # self.q_model_kwargs = dict()
        super().__init__(ModelCls=ModelCls, QModelCls=QModelCls, **kwargs)

    def to_device(self, cuda_idx=None):
        BaseAgent.to_device(self, cuda_idx)
        self.target_model.to(self.device)
        # self.pi_head.to(self.device)
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        self.target_q1_model.to(self.device)
        self.target_q2_model.to(self.device)

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None  # Don't let base agent try to load.
        # import pdb; pdb.set_trace()
        BaseAgent.initialize(self, env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.initial_model_state_dict = _initial_model_state_dict

        env_model_kwargs = dict(
            # observation_shape=self.shared_layer_size,
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )
        self.action_size=env_spaces.action.shape[0]
        # self.pi_head = MlpModel(self.shared_layer_size, [256, 256], 2 * self.action_size)
        # self.pi_head = MlpModel(np.prod(env_spaces.observation.shape.state), [256, 256], 2 * self.action_size)
        # self.pi_head = PiMlpModel(observation_shape=env_spaces.observation.shape.state, action_size=env_spaces.action.shape[0]
                # , hidden_sizes=[256, 256])
        
        #self.q1_model = self.QModelCls(**env_model_kwargs, **self.q_model_kwargs)
        #self.q2_model = self.QModelCls(**env_model_kwargs, **self.q_model_kwargs)
        #self.target_q1_model = self.QModelCls(**env_model_kwargs,
        #    **self.q_model_kwargs)
        #self.target_q2_model = self.QModelCls(**env_model_kwargs,
        #    **self.q_model_kwargs)

        #q_model_kwargs = dict(input_size=np.prod(env_spaces.observation.shape.state) + self.action_size, hidden_sizes=[256, 256],
        #                      output_size=1)
        q_model_kwargs = dict(input_size=self.shared_layer_size + self.action_size, hidden_sizes=[256, 256],
                              output_size=1)
        self.q1_model = MlpModel(**q_model_kwargs)
        self.q2_model = MlpModel(**q_model_kwargs)
        self.target_q1_model = MlpModel(**q_model_kwargs)
        self.target_q2_model = MlpModel(**q_model_kwargs)
        self.target_model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)

        self.target_q1_model.load_state_dict(self.q1_model.state_dict())
        self.target_q2_model.load_state_dict(self.q2_model.state_dict())

        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )
    def all_q(self, model_input: AgentInputs, init_rnn_states: QRnnState, action=None) -> (QValues, QRnnState):
        """Compute twin Q-values for state/observation and input action
        (with grad)."""
        model_input = buffer_to(model_input, device=self.device)
        features, rnn_state = self.model(*model_input, None)

        q1_rnn_state, q2_rnn_state = buffer_to((init_rnn_states.q1, init_rnn_states.q2), self.device)
        target_q1_rnn_state, target_q2_rnn_state = buffer_to((init_rnn_states.target_q1, init_rnn_states.target_q2),
                                                             self.device)
        action = buffer_to(action, device=self.device)
        q1 = self.q1_model(torch.cat((features, action), dim=-1)).squeeze(-1)
        q2 = self.q2_model(torch.cat((features, action), dim=-1)).squeeze(-1)
        with torch.no_grad():
            target_q1 = self.target_q1_model(torch.cat((features, action), dim=-1)).squeeze(-1)
            target_q2 = self.target_q2_model(torch.cat((features, action), dim=-1)).squeeze(-1)

        q_values = QValues(q1=q1, q2=q2, target_q1=target_q1, target_q2=target_q2)
        rnn_states = QRnnState(q1=q1_rnn_state, q2=q2_rnn_state, target_q1=target_q1_rnn_state,
                               target_q2=target_q2_rnn_state)
        q_values = buffer_to(q_values, device='cpu')
        return q_values, rnn_states


    def features(self, model_input, rnn_state):
        features, rnn_state = self.model.base_model_forward(*model_input, rnn_state)
        return features

    def target_features(self, model_input, rnn_state):
        target_features, rnn_state = self.target_model.base_model_forward(*model_input, rnn_state)
        return target_features


    def forward(self, model_input, old_action, rnn_state):
        model_input, old_action, rnn_state = buffer_to((model_input, old_action, rnn_state), device=self.device)
        features = self.features(model_input, rnn_state)
        # mean, log_std = self.model.pi_head_forward(features)


        # features, mean, log_std, rnn_state = self.model(*model_input, rnn_state)
        # dist_info = DistInfoStd(mean=mean, log_std=log_std)
        # new_action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        new_action, log_pi, dist_info, rnn_state = self.pi(features, model_input, rnn_state)

        # q_features, _ = self.q_base_model(*model_input, rnn_state)
        # q_features = features
        q_features = self.features(model_input, rnn_state)

        q1 = self.q1_model(torch.cat((q_features, old_action), dim=-1)).squeeze(-1)
        q2 = self.q2_model(torch.cat((q_features, old_action), dim=-1)).squeeze(-1)

        new_q1 = self.q1_model(torch.cat((q_features.detach(), new_action), dim=-1)).squeeze(-1)
        new_q2 = self.q2_model(torch.cat((q_features.detach(), new_action), dim=-1)).squeeze(-1)

        multi_head_output = dict(q1=q1,
                                 q2=q2,
                                 new_q1=new_q1,
                                 new_q2=new_q2,
                                 new_action=new_action,
                                 log_pi=log_pi,
                                 pi_mean=dist_info.mean,
                                 pi_log_std=dist_info.log_std)
        # for key, value in multi_head_output.items():
            # multi_head_output[key] = value.cpu()
        return multi_head_output


    def q(self, features, model_input, q1_rnn_state, q2_rnn_state, action):
        model_input = buffer_to(model_input, device=self.device)
        q1_rnn_state, q2_rnn_state = buffer_to((q1_rnn_state, q2_rnn_state), self.device)
        action = buffer_to(action, device=self.device)
        q1 = self.q1_model(torch.cat((features, action), dim=-1)).squeeze(-1)
        q2 = self.q2_model(torch.cat((features, action), dim=-1)).squeeze(-1)
        return q1, q2

    def target_q(self, features, model_input, q1_rnn_state, q2_rnn_state, action=None):
        model_input = buffer_to(model_input, device=self.device)
        q1_rnn_state, q2_rnn_state = buffer_to((q1_rnn_state, q2_rnn_state), self.device)
        action = buffer_to(action, device=self.device)

        q1 = self.target_q1_model(torch.cat((features.detach(), action), dim=-1)).squeeze(-1)
        q2 = self.target_q2_model(torch.cat((features.detach(), action), dim=-1)).squeeze(-1)
        return q1, q2

    def pi(self, features, model_input, init_rnn_state):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_input = buffer_to(model_input, device=self.device)
        rnn_state = buffer_to(init_rnn_state, self.device)
        mean, log_std = self.model.pi_head_forward(features)

        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        return action, log_pi, dist_info, rnn_state  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        pi_rnn_state, q1_rnn_state, q2_rnn_state = self.decode_rnn_state()
        features, mean, log_std, rnn_state = self.model(*model_inputs, pi_rnn_state)
        lead_dim, T, B, _ = infer_leading_dims(model_inputs[0].state, 1)
        q1_rnn_state = torch.zeros(1, B, 1)
        q2_rnn_state = torch.zeros(1, B, 1)

        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)

        new_rnn_state = RnnState(pi=pi_rnn_state, q1=q1_rnn_state, q2=q2_rnn_state)
        prev_rnn_state = self.prev_rnn_state or buffer_func(new_rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case, model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        prev_rnn_state = buffer_to(prev_rnn_state, device="cpu")
        agent_info = AgentInfo(dist_info=dist_info, prev_rnn_state=prev_rnn_state)

        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(new_rnn_state)
        return AgentStep(action=action, agent_info=agent_info)

    def decode_rnn_state(self):
        if self.prev_rnn_state is None:
            return None, None, None
        else:
            return self.prev_rnn_state.pi, self.prev_rnn_state.q1, self.prev_rnn_state.q2
    
    def pi_head_parameters(self):
        return self.model.pi_head_parameters()

    def base_model_parameters(self):
        return self.model.base_model_parameters()

    def q1_parameters(self):
        return self.q1_model.parameters()

    def q2_parameters(self):
        return self.q2_model.parameters()

    def update_target(self, tau=1):
        update_state_dict(self.target_q1_model, self.q1_model.state_dict(), tau)
        update_state_dict(self.target_q2_model, self.q2_model.state_dict(), tau)
        update_state_dict(self.target_model, self.model.state_dict(), tau)

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )
