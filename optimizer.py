from __future__ import division
import numpy as np
import torch


class TransformerAdamTrainer(object):
    """
    Proposed in the paper "Attention is all you need" (https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [Page 7, Eq. 3]
    In this the learning rate of Adam Optimizer is increased for the first warmup steps followed by a gradual decay.

    In the paper, warmup steps = 4000, beta2=0.98

    """
    def __init__(self, model, alpha=1.0, dim=512, warmup_steps=16000, beta_1=0.9, beta_2=0.997, eps=1e-9, lr_constant=2.0):
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=alpha,
                                          betas=(beta_1, beta_2),
                                          eps=eps)
        self.dim = dim
        self.warmup_steps = warmup_steps
        self.learning_rate_constant = lr_constant
        self.steps = 0

    def step(self):
        self.steps += 1
        decay = (self.dim ** (-0.5)) * np.min([self.steps ** (-0.5), self.steps * (self.warmup_steps ** (-1.5))])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate_constant * decay
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
