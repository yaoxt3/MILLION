import torch
import time
import numpy as np
# from compressive_transformer_pytorch import CompressiveTransformer as CompressiveTransformerPyTorch
# from compressive_transformer_pytorch import CompressiveTransformer as CompressiveTransformerPyTorch
from learning_to_be_taught.models.compressive_transformer_pytorch import CompressiveTransformerPyTorch
from compressive_transformer_pytorch.compressive_transformer_pytorch import Memory
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
# from learning_to_be_taught.models.gated_transformer import GatedTransformer, SIZES
from learning_to_be_taught.models.transformer_models import generate_square_subsequent_mask
from learning_to_be_taught.models.transformer_xl.mem_transformer import MemTransformerLM


State = namedarraytuple("State", ["sequence", "memory", 'compressed_memory', 'length'])#, 'memory_filled', 'compressed_memory_filled'])

SIZES = dict(
    zero=dict(dim=64,
              cmem_ratio=4,
              num_heads=4,
              depth=0),
    tiny=dict(dim=64,
              cmem_ratio=4,
              num_heads=4,
              depth=1),
    small=dict(dim=64,
               cmem_ratio=4,
               num_heads=4,
               depth=3),
    medium=dict(dim=64,
                cmem_ratio=4,
                num_heads=4,
                depth=6),
    large=dict(dim=1024,
               cmem_ratio=4,
               num_heads=12,
               depth=12)
)


class TransformerXL(torch.nn.Module):
    def __init__(self,
                 observation_shape,
                 action_size,
                 linear_value_output=True,
                 sequence_length=64,
                 seperate_value_network=True,
                 size='medium',
                 ):
        super().__init__()
        self.state_size = np.prod(observation_shape.state)
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.transformer_dim = SIZES[size]['dim']
        self.depth = SIZES[size]['depth']
        self.cmem_ratio = SIZES[size]['cmem_ratio']
        self.cmem_length = self.sequence_length // self.cmem_ratio
        memory_layers = range(1, self.depth + 1)
        self.transformer = MemTransformerLM(n_token=20000,
                                            n_layer=SIZES[size]['depth'],
                                            n_head=SIZES[size]['num_heads'],
                                            d_model=self.transformer_dim,
                                            d_head=self.transformer_dim,
                                            d_inner=self.transformer_dim,
                                            dropout=0.0,
                                            dropatt=0.0,
                                            pre_lnorm=True)


        self.transformer.token_emb = torch.nn.Identity()  # don't use token embedding in compressive transforrmer
        self.transformer.to_logits = torch.nn.Identity()
        # self.input_layer_norm = torch.nn.LayerNorm(self.state_size)
        self.input_layer_norm = torch.nn.Identity()
        # self.output_layer_norm = torch.nn.LayerNorm(self.transformer_dim)
        self.output_layer_norm = torch.nn.Identity()
        self.softplus = torch.nn.Softplus()
        self.pi_head = MlpModel(input_size=self.transformer_dim,
                                hidden_sizes=[256, ],
                                output_size=2 * action_size)
        self.value_head = MlpModel(input_size=self.transformer_dim,
                                   hidden_sizes=[256, ],
                                   output_size=1 if linear_value_output else None)
        self.mask = torch.ones((self.sequence_length, self.sequence_length), dtype=torch.int8).triu()

    def forward(self, observation, prev_action, prev_reward, state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        # print(f'step in episode {observation.step_in_episode}')
        aux_loss = None
        if T == 1:
            transformer_output, state = self.sample_forward(observation.state, state)
            value = torch.zeros(B)
        elif T == self.sequence_length:
            transformer_output, aux_loss = self.optim_forward(observation.state, state)
            value = self.value_head(transformer_output).reshape(T * B, -1)
        else:
            raise NotImplementedError

        pi_output = self.pi_head(transformer_output).view(T * B, -1)
        mu, std = pi_output[:, :self.action_size], pi_output[:, self.action_size:]
        std = self.softplus(std)
        mu = torch.tanh(mu)

        # pi_output = self.softplus(pi_output * 1) + 1
        # mu, std = pi_output[:, :self.action_size], pi_output[:, self.action_size:]


        mu, std, value = restore_leading_dims((mu, std, value), lead_dim, T, B)
        return mu, std, value, state, aux_loss

    def sample_forward(self, observation, state):
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)
        observation = self.input_layer_norm(observation)

        device = observation.device
        if state is None:
            observations = torch.zeros((self.sequence_length, B, self.state_size), device=device)
            length = torch.zeros((1, B, 1), device=device, dtype=torch.int64)
            memory = torch.zeros((self.depth, B, self.sequence_length, self.transformer_dim), device=device)
            compressed_memory = torch.zeros((self.depth, B, self.cmem_length, self.transformer_dim), device=device)
        else:
            observations = state.sequence
            length = state.length.clamp_max(self.sequence_length - 1)
            memory = state.memory
            compressed_memory = state.compressed_memory

        # write new observations in tensor with older observations
        observation = observation.view(B, -1)
        # observations = torch.cat((observations, observation), dim=0)[1:]
        indexes = tuple(torch.cat((length[0, :], torch.arange(B, device=device).unsqueeze(-1)), dim=-1).t())
        observations.index_put_(indexes, observation)

        transformer_output, new_memory, _ = self.transformer(observations.transpose(0, 1), Memory(mem=memory, compressed_mem=None))
        transformer_output = self.output_layer_norm(transformer_output).transpose(0,1)
        # output = transformer_output.transpose(0, 1)[-1]
        output = transformer_output[length[0, :, 0], torch.arange(B)]
        length = torch.fmod(length + 1, self.sequence_length)

        reset = (length == 0).int()[0, :, 0].reshape(B, 1, 1, 1).transpose(0, 1).expand_as(memory)
        # print(f'length {length[0, :, 0]}')
        # if B > 1 and length[0, 0, 0] != length[0, 1, 0]:
        #     breakpoint()
        memory = reset * new_memory.mem + (1 - reset) * memory

        state = State(sequence=observations, length=length, memory=memory, compressed_memory=compressed_memory)
        return output, state

    def optim_forward(self, observation, state):
        observation = self.input_layer_norm(observation)
        output, _, aux_loss = self.transformer(observation.transpose(0, 1), Memory(mem=state.memory, compressed_mem=None))
        output = self.output_layer_norm(output)
        return output.transpose(0, 1), aux_loss
