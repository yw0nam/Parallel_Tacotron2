# %%
from models.blocks import LightweightConv, LinearNorm, LConvBlock, ConvNorm1D, SwishBlock, ConvBlock
import numpy as np
import torch.nn as nn
from models.modules import ResidualEncoder, TextEncdoer, DurationPredictor
from configs import *
from dataset import Transformer_Collator, TTSdataset
from torch.utils.data import DataLoader
import torch
from utils import get_mask_from_lengths
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
text_mask = input_['pos_text'].lt(1)
batch_size = text_mask.size(0)
# %%
pred_mel_len = torch.round(dur.sum(-1)).type_as(input_['pos_text'])
pred_mel_len = torch.clamp(pred_mel_len, max=1000)
pred_max_mel_len = pred_mel_len.max().item()
pred_mel_mask = get_mask_from_lengths(pred_mel_len, pred_max_mel_len)
# %%
text_mask_ = text_mask.unsqueeze(1).expand(-1, pred_mel_mask.shape[1], -1) # [B, tat_len, src_len]
pred_mel_mask_ = pred_mel_mask.unsqueeze(-1).expand(-1, -1, text_mask.shape[1])
attn_mask = torch.zeros((text_mask.shape[0], pred_mel_mask.shape[1], text_mask.shape[1])).type_as(input_['pos_text'])

attn_mask = attn_mask.masked_fill(text_mask_, 1.)
attn_mask = attn_mask.masked_fill(pred_mel_mask_, 1.)
attn_mask = attn_mask.bool()
# %%
e_k = torch.cumsum(dur, dim=1)
s_k = e_k - dur
e_k = e_k.unsqueeze(1).expand(batch_size, pred_max_mel_len, -1)
s_k = s_k.unsqueeze(1).expand(batch_size, pred_max_mel_len, -1)
# %%
t_arange = torch.arange(1, pred_max_mel_len+1).type_as(input_['pos_text']).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, attn_mask.size(2)
        )
# %%
S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(attn_mask, 0)
# %%
conv_c = ConvBlock(
    model_config.r_out_size,
    8,
    0,
    3,
    nn.SiLU()
)

# %%
temp = conv_c(v)
# %%
conv_c = ConvBlock(
    model_config.r_out_size,
    model_config.l_conv_out_size,
    0,
    model_config.l_conv_kernel_size,
    nn.SiLU()
)

swish_c = SwishBlock(
    model_config.l_conv_out_size+2,
    model_config.l_swish_c_out_size,
    model_config.l_swish_c_out_size,
)

conv_w = ConvBlock(
    model_config.r_out_size,
    model_config.l_conv_out_size,
    0,
    model_config.l_conv_kernel_size,
    nn.SiLU()
)

swish_w = SwishBlock(
    model_config.l_conv_out_size+2,
    model_config.l_swish_w_out_size,
    model_config.l_swish_w_out_size,
)

proj_w = LinearNorm(model_config.l_swish_w_out_size, 1)
softmax_w = nn.Softmax(dim=2)

linear_einsum = LinearNorm(
    model_config.l_swish_c_out_size, model_config.r_out_size)  # A

# %%
W = swish_w(S, E, conv_w(v)) # [B, T, K, dim_w]
# %%
W = proj_w(W).squeeze(-1).masked_fill(text_mask_, -np.inf) #[B, T, K]
# %%
W = softmax_w(W) #[B, T, K]
# %%
W = W.masked_fill(pred_mel_mask_, 0.)
# %%
C = swish_c(S, E, conv_c(v))

# %%
C.shape
# %%
torch.einsum('btk,btkp->btp', W, C).shape

# %%
W.shape
# %%
C.shape
# %%
