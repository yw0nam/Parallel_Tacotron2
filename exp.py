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
text_encoder = TextEncdoer(model_config, 130)
encoder= ResidualEncoder(model_config, 80)
duration_predictor = DurationPredictor(model_config)
speaker_emb_layer = nn.Embedding(2, 64, 0)
speaker_emb = speaker_emb_layer(torch.LongTensor([0, 0, 0, 1]))
# %%
text_out = text_encoder(input_['text'], input_['pos_text'], input_['pos_text'].lt(1))
x, attn, mu, log_var = encoder(text_out, input_['pos_text'], input_['pos_text'].lt(1),
                       input_['mel'], input_['pos_mel'], input_['pos_mel'].lt(1),
                       speaker_emb)

# %%
v, dur = duration_predictor(x, input_['pos_text'].lt(1))
# %%
learned_upsampling = LearnedUpsampling(model_config)
# %%
upsampled_rep, pred_mel_mask, pred_mel_len, W = learned_upsampling(dur, v, input_['pos_text'], input_['pos_text'].lt(1))
# %%
decoder = Decoder(model_config)
# %%
mel_iters, mask = decoder(upsampled_rep, pred_mel_mask)
# %%
lconv_blocks = nn.ModuleList(
    [
        LConvBlock(
            hidden_size=32,
            kernel_size=model_config.dc_lconv_kernel_size,
            num_heads=model_config.n_head,
            dropout=model_config.dc_dropout_p,
            stride=1
        )
        for _ in range(model_config.dc_layer_num)
    ]
)
projection = nn.ModuleList(
    [
        LinearNorm(
            32, 80
        )
        for _ in range(model_config.dc_layer_num)
    ]
)

# %%
x = upsampled_rep.masked_fill(pred_mel_mask.unsqueeze(-1), 0)

# %%
x = torch.tanh(lconv_blocks[0](
                x, mask=pred_mel_mask
            ))
# %%
x.shape
# %%
