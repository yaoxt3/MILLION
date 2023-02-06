"""
copy from https://github.com/aluscher/torchbeastpopart/blob/master/torchbeast/core/popart.py

added support for multi-task learning
"""
import math
import torch


class PopArtLayer(torch.nn.Module):

    def __init__(self, input_features=256, output_features=1, beta=1e-4):
        self.beta = beta

        super(PopArtLayer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = torch.nn.Parameter(torch.Tensor(output_features))

        self.register_buffer('mu', torch.zeros(output_features, requires_grad=False))
        self.register_buffer('sigma', torch.ones(output_features, requires_grad=False))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, task=None):
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(-1)
        input_shape = inputs.shape
        inputs = inputs.reshape(-1, self.input_features)

        normalized_output = inputs.mm(self.weight.t())
        normalized_output += self.bias.unsqueeze(0).expand_as(normalized_output)
        normalized_output = normalized_output.reshape(*input_shape[:-1], self.output_features)

        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        if task is not None:
            output = output.gather(-1, task.unsqueeze(-1))
            normalized_output = normalized_output.gather(-1, task.unsqueeze(-1))

        return [output.squeeze(-1), normalized_output.squeeze(-1)]

    @torch.no_grad()
    def normalize(self, inputs, task=None):
        """
        task: task ids
        """
        task = torch.zeros(inputs.shape, dtype=torch.int64) if task is None else task
        input_device = inputs.device
        inputs = inputs.to(self.mu.device)
        mu = self.mu.expand(*inputs.shape, self.output_features).gather(-1, task.unsqueeze(-1)).squeeze(-1)
        sigma = self.sigma.expand(*inputs.shape, self.output_features).gather(-1, task.unsqueeze(-1)).squeeze(-1)
        output = (inputs - mu) / sigma
        return output.to(input_device)

    @torch.no_grad()
    def update_parameters(self, vs, task):
        """
        task: one hot vector of tasks
        """
        vs, task = vs.to(self.mu.device), task.to(self.mu.device)

        oldmu = self.mu
        oldsigma = self.sigma

        vs = vs * task
        n = task.sum((0, 1))
        mu = vs.sum((0, 1)) / n
        nu = torch.sum(vs ** 2, (0, 1)) / n
        sigma = torch.sqrt(nu - mu ** 2)
        sigma = torch.clamp(sigma, min=1e-2, max=1e+6)

        mu[torch.isnan(mu)] = self.mu[torch.isnan(mu)]
        sigma[torch.isnan(sigma)] = self.sigma[torch.isnan(sigma)]

        self.mu = (1 - self.beta) * self.mu + self.beta * mu
        self.sigma = (1 - self.beta) * self.sigma + self.beta * sigma
        # print(f'new sigma: {self.sigma}#################################################3')

        self.weight.data = (self.weight.t() * oldsigma / self.sigma).t()
        self.bias.data = (oldsigma * self.bias + oldmu - self.mu) / self.sigma

    def state_dict(self):
        return dict(mu=self.mu,
                    sigma=self.sigma,
                    weight=self.weight.data,
                    bias=self.bias.data)

    def load_state_dict(self, state_dict):
        with torch.no_grad():
            self.mu = state_dict['mu']
            self.sigma = state_dict['sigma']
            self.weight.data = state_dict['weight']
            self.bias.data = state_dict['bias']
