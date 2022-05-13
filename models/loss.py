
import torch
import torch.nn as nn
from .soft_dtw_cuda import SoftDTW


class ParallelTacotron2Loss(nn.Module):
    """ Parallel Tacotron 2 Loss """

    def __init__(self, model_config, train_config):
        super(ParallelTacotron2Loss, self).__init__()
        
        self.sdtw_loss = SoftDTW(
            use_cuda=True,
            gamma=train_config.sdtw_gamma,
            warp=train_config.sdtw_warp,
        )
        self.mae_loss = nn.L1Loss(reduction='none')
