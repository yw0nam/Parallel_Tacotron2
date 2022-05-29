# %%
import numpy as np
import torch.nn as nn
from configs import *
from dataset import Transformer_Collator, TTSdataset
from torch.utils.data import DataLoader
import torch
from models.loss import ParallelTacotron2Loss
from models.soft_dtw_cuda import SoftDTW
from models.pl_model import PL_model
# %%
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# %%
config = DataConfig(
    root_dir="/data1/spow12/datas/TTS/LJSpeech-1.1/",
    train_csv="metadata_train.csv",
    val_csv="metadata_val.csv"
)
# %%
dataset = TTSdataset(config)
train_loader = DataLoader(dataset, 8, num_workers=8,
                            collate_fn=Transformer_Collator(), 
                            pin_memory=True,shuffle=False)

model_config = ModelConfig()
train_config = TrainConfig()
# %%
model = PL_model(
    train_config,
    model_config,
    config.vocab_size,
    config.n_mels,
    config.speaker_num
).cuda()
# %%
for i in tqdm(train_loader):
    input_, label = i
    for key in input_:
        input_[key] = input_[key].cuda()
    for key in label:
        label[key] = label[key].cuda()
    out = model(input_)
    break
# %%
out
# %%
from models.loss import ParallelTacotron2Loss
pt_loss = ParallelTacotron2Loss(train_config)
# %%
with torch.autocast(device_type='cuda'):
    loss = pt_loss(out, label, 0)
# %%
loss
# %%
