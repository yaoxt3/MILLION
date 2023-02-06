import torch
from torch.optim.optimizer import Optimizer
import numpy as np
from torch.utils.cpp_extension import load


@torch.no_grad()
def pcgrad_py(grads, mopt_wei):
    grads_target = grads.detach().clone()
    ngrads = grads.shape[0]
    for i in np.random.permutation(ngrads):
        for j in np.random.permutation(ngrads):
            if (j == i): continue
            cth = (grads[i] * grads[j]).sum()
            # norm_i = torch.norm(grads[i])
            norm_j = torch.norm(grads[j])
            if (cth < 0).item():
                cth.mul_(- mopt_wei / (norm_j ** 2))
                grads_target[i].add_(cth,
                                     grads[j])
    d_p = grads_target.sum(dim=0)
    return d_p


class PCGradSGD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, args=None, mopt_wei=1, ntask=3
                 ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        self.mopt_wei = mopt_wei # args.mopt_wei
        self.inited = False
        self.ntask = ntask # args.nBlocks
        self.rgem = pcgrad_py

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def virtual_init(self):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                state['shape'] = p.shape
                state['ngrad'] = self.ntask
                state['now_idx'] = 0
                state["grads"] = torch.stack(
                    [torch.zeros_like(p.data) for _ in range(
                        self.ntask
                    )], dim=0
                )
                print(state['ngrad'], state['now_idx'], state['shape'])
        self.inited = True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None or p.grad.abs().sum() < 1e-6:
                    continue
                state = self.state[p]
                d_p = p.grad.data
                state['grads'][state['now_idx']].add_(d_p)
                state['now_idx'] += 1

    def step2(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            ttl = cnt = 0
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                ngrads = state["now_idx"]
                if ngrads == 0: continue
                grads = state['grads'][:ngrads]
                ttl += ngrads * (ngrads - 1)

                d_p = self.rgem(grads, self.mopt_wei)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

                state['grads'].zero_()
                state['now_idx'] = 0

        return ttl, cnt
