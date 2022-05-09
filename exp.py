# %%
from models.blocks import LightweightConv, LinearNorm, LConvBlock, ConvNorm1D
import numpy as np
import torch.nn as nn
from models.modules import ResidualEncoder, TextEncdoer
from configs import *
from dataset import Transformer_Collator, TTSdataset
from torch.utils.data import DataLoader
import torch
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
                            pin_memory=True,shuffle=False)
# %%
for i in train_loader:
    input_, label = i
    break
# %%
model_config = ModelConfig()

# %%
text_encoder = TextEncdoer(model_config, 130)
encoder= ResidualEncoder(model_config, 80)
# %%
text_out = text_encoder(input_['text'], input_['pos_text'])
# %%
speaker_emb_layer = nn.Embedding(2, 64, 0)
speaker_emb = speaker_emb_layer(torch.LongTensor([0, 0, 0, 1]))
# %%
x, attn, mu, log_var = encoder(text_out, input_['pos_text'],
                       input_['mel'], input_['pos_mel'],
                       speaker_emb)
# %%
x.shape
# %%
x
# %%
lconv_blocks = nn.ModuleList(
    [
        LConvBlock(
            hidden_size=32,
            kernel_size=3,
            num_heads=8,
            dropout=0.1,
            stride=1
        )
        for _ in range(4)
    ]
)

# %%
temp = x
for block in lconv_blocks:
    temp = block(temp, input_['pos_text'].lt(1))
# %%
duration_out = nn.Sequential(
    LinearNorm(32, 1),
    nn.Softplus()
)
# %%
d = duration_out(temp)
# %%

# %%
x.shape
# %%
temp.masked_fill(input_['pos_text'].lt(1).unsqueeze(2), 0)
# %%
d.shape
# %%
