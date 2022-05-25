
import torch
import torch.nn as nn
from .soft_dtw_cuda import SoftDTW


class ParallelTacotron2Loss(nn.Module):
    """ Parallel Tacotron 2 Loss """

    def __init__(self, train_config):
        super(ParallelTacotron2Loss, self).__init__()
        self.lambda_ = train_config.loss_lambda
        self.start = train_config.kl_start
        self.end = train_config.kl_end
        self.upper = train_config.kl_upper

        self.sdtw_loss = SoftDTW(
            use_cuda=True,
            gamma=train_config.gamma,
            warp=train_config.warp
        )
        self.mae_loss = nn.L1Loss()

    def kl_anneal(self, step):
        if step < self.start:
            return .0
        elif step >= self.end:
            return self.upper
        else:
            return self.upper*((step - self.start) / (self.end - self.start))

    def forward(self, prediction, label, step):
        
        mel, mel_length, text_length = label.values()
        mel_iters, dur, mus, log_vars, attn, W = prediction.values()
        mel.requires_grad = False
        text_length.requires_grad = False
        mel_length.requires_grad = False

        # Iterative Loss Using Soft-DTW
        mel_iter_loss = torch.zeros_like(
            mel_length, dtype=mel.dtype)
        for mel_iter in mel_iters:
            mel_iter_loss += self.sdtw_loss(mel_iter, mel)
        mel_loss = (mel_iter_loss / (len(mel_iters) * mel_length)).mean()

        # Duration Loss
        duration_loss = self.lambda_ * (self.mae_loss(dur.sum(-1), mel_length))

        # KL Divergence Loss
        beta = torch.tensor(self.kl_anneal(step))
        kl_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())

        total_loss = (
            mel_loss + duration_loss + beta * kl_loss
        )

        return {
            'total_loss' : total_loss,
            'mel_loss' : mel_loss,
            'duration_loss' : duration_loss,
            'kl_loss' : kl_loss,
            'beta' : beta,
        }
