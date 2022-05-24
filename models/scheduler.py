import numpy as np
import torch

class ScheduledOptim(torch.optim.lr_scheduler._LRScheduler):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, train_config):
        self.optimizer = optimizer
        self.warmup_step = train_config.warmup_step
        self.anneal_steps = train_config.anneal_steps
        self.anneal_rate = train_config.anneal_rate
        self.lr = train_config.lr
        self.update_steps = 1
        super().__init__(optimizer)
        
    def update_lr(self):
        lr = np.min(
            [
                np.power(self.update_steps, -0.5),
                np.power(self.warmup_step, -1.5) * self.update_steps,
            ]
        )
        for s in self.anneal_steps:
            if self.update_steps > s:
                lr = lr * self.anneal_rate
        return lr

    def set_lr(self, optimizer, lr):
        """ Learning rate scheduling per step """

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
    def step(self):
        lr = self.update_lr()
        self.set_lr(self.optimizer, lr)
        self.lr = lr
        self.update_steps += 1
        return self.lr