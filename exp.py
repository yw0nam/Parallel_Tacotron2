# %%
from models.blocks import LightweightConv, LinearNorm, LConvBlock, ConvNorm1D, SwishBlock, ConvBlock
import numpy as np
import torch.nn as nn
from models.modules import ResidualEncoder, TextEncdoer, DurationPredictor, LearnedUpsampling, Decoder
from configs import *
from dataset import Transformer_Collator, TTSdataset
from torch.utils.data import DataLoader
import torch
from utils import get_mask_from_lengths, get_sinusoid_encoding_table
from models.soft_dtw_cuda import SoftDTW
from models.parallel_tacotron2 import ParallelTacotron2
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
# %%
model = ParallelTacotron2(model_config, 130, 80, 1)
# %%
out = model(**input_, speaker=torch.LongTensor([0, 0, 0, 0]))