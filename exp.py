# %%
from models.blocks import LightweightConv, LinearNorm, LConvBlock, ConvNorm1D
import numpy as np
import torch.nn as nn
from models.modules import ResidualEncoder, TextEncdoer
from configs import *
from dataset import Transformer_Collator, TTSdataset
from torch.utils.data import DataLoader
# %%
from utils import get_sinusoid_encoding_table
# %%
config = DataConfig(
    root_dir="/data1/spow12/datas/TTS/LJSpeech-1.1/",
    train_csv="metadata_train.csv",
    val_csv="metadata_val.csv"
)
# %%
dataset = TTSdataset(config)
# %%
train_loader = DataLoader(dataset, 4, num_workers=8,
                            collate_fn=Transformer_Collator(), 
                            pin_memory=True,shuffle=True)
# %%
for i in train_loader:
    input_, label = i
    break
# %%
model_config = ModelConfig()

# %%
text_encoder = TextEncdoer(model_config, 130)
encoder= ResidualEncoder(model_config)
# %%
text_out = text_encoder(input_['text'], input_['pos_text'])
residual_out = encoder(input_['mel'], input_['pos_mel'])
# %%
text_out.shape
# %%
residual_out.shape
# %%
pos_emb_layer = nn.Embedding.from_pretrained(
    get_sinusoid_encoding_table(
        model_config.max_seq_len+1,
        model_config.r_hidden_size[0],
        padding_idx=0
    ),
    freeze=True
)
pos_emb_layer.requires_grad_ = False
# %%
pos_mel_emb = pos_emb_layer(input_['pos_mel'])
# %%
pos_mel_emb.shape
# %%
