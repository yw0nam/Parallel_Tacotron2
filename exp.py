# %%
import numpy as np
import torch.nn as nn
from configs import *
from dataset import Transformer_Collator, TTSdataset
from torch.utils.data import DataLoader
import torch
from models.soft_dtw_cuda import SoftDTW
from models.pl_model import PL_model
# %%
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# %%
config = DataConfig(
    root_dir="/data1/spow12/datas/TTS/LJSpeech-1.1/",
    train_csv="metadata_train.csv",
    val_csv="metadata_val.csv"
)
# %%
dataset = TTSdataset(config)
train_loader = DataLoader(dataset, 4, num_workers=8,
                            collate_fn=Transformer_Collator(), 
                            pin_memory=True,shuffle=False)
for i in train_loader:
    input_, label = i
    break
# %%
model_config = ModelConfig()
train_config = TrainConfig()
# %%
model = PL_model(
    train_config, 
    model_config,
    config.vocab_size,
    config.n_mels,
    config.speaker_num
)
# %%
out = model(input_)
# %%
out['W'].size()
# %%
plt.imshow(out['attn'].detach()[0].numpy())
# %%
fig = plt.figure(figsize=(25, 15))
for i in range(1, 5):
    ax = fig.add_subplot(4, 1, i)
    ax.imshow(out['attn'].detach()[i-1])
    ax.set_title("attention_%d"%i)

# %%
writer = SummaryWriter()
writer.add_figure('attention', fig, 0)
writer.add_image('attn_2', out['attn'][1], 0, dataformats='HW')
writer.add_image('attn_3', out['attn'][2], 0, dataformats='HW')
writer.add_image('attn_4', out['attn'][3], 0, dataformats='HW')


# %%
