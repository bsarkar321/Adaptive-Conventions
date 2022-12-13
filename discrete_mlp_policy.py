from garage.torch.policies.stochastic_policy import StochasticPolicy

from garage.torch.modules.mlp_module import MLPModule

from torch.distributions.independent import Independent

import torch
from torch import nn

from torch.nn import functional as F


class DiscreteMLPPolicy(StochasticPolicy):
    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='GDiscreteMLPPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = MLPModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

    def forward(self, observations):
        observations = observations.reshape(
            -1, *self._env_spec.observation_space.shape)
        logits = self._module(observations)
        # logits = torch.softmax(output, dim=1)
        # print(output, logits)
        dist = torch.distributions.Categorical(logits=logits)
        # print(dist)
        # dist = Independent(dist, 1)
        return dist, {}
