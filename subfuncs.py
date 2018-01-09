from __future__ import division

import numpy as np
from chainer.training import extension
import torch


class TransformerAdamTrainer(object):
    """
  Proposed in the paper "Attention is all you need" (https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [Page 7, Eq. 3]
  In this the learning rate of Adam Optimizer is increased for the first warmup steps followed by a gradual decay
  """

    def __init__(self, model, alpha=1.0, dim=512, warmup_steps=4000, beta_1=0.9, beta_2=0.98, eps=1e-9):

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=alpha,
                                          betas=(beta_1, beta_2),
                                          eps=eps)
        self.dim = dim
        self.warmup_steps = warmup_steps
        self.steps = 0

    def step(self):
        self.steps += 1
        decay = (self.dim ** (-0.5)) * np.min([self.steps ** (-0.5), self.steps * (self.warmup_steps ** (-1.5))])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1. * decay

        self.optimizer.step()

        if self.steps % 200 == 0:
            print('> Optimizer Logging')
            print('  Steps=%d, learning_rate=%.2e' % (self.steps, decay))

    def zero_grad(self):
        self.optimizer.zero_grad()


class VaswaniRule(extension.Extension):
    """Trainer extension to shift an optimizer attribute magically by Vaswani.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, d, warmup_steps=4000,
                 init=None, target=None, optimizer=None,
                 scale=1.):
        self._attr = attr
        self._d_inv05 = d ** (-0.5) * scale
        self._warmup_steps_inv15 = warmup_steps ** (-1.5)
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            # self._init = getattr(optimizer, self._attr)
            self._init = self._d_inv05 * (1. * self._warmup_steps_inv15)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        self._t += 1

        optimizer = self._get_optimizer(trainer)
        value = self._d_inv05 * \
                min(self._t ** (-0.5), self._t * self._warmup_steps_inv15)
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
