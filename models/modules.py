from .blocks import ConvBlock, TransformerEncoder, LConvBlock, ConvNorm1D, LinearNorm
import torch
from torch import nn
from utils import get_sinusoid_encoding_table

class TextEncdoer(nn.Module):
    def __init__(self, model_config, vocab_size, activation=nn.ReLU()):
        super(TextEncdoer, self).__init__()
        
        self.text_emb_layer = nn.Embedding(vocab_size ,model_config.t_embedding_size)
        
        self.pos_emb_layer = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                model_config.max_seq_len+1,
                model_config.t_hidden_size,
                padding_idx=0
            ),
            freeze=True
        )
        self.pos_emb_layer.requires_grad_ = False
        
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channel=model_config.t_hidden_size, 
                    out_channel=model_config.t_hidden_size, 
                    dropout=model_config.t_dropout_p, 
                    activation=activation
                )
                for _ in range(model_config.t_conv_num)
            ]
        )

        self.TF_encoder = nn.ModuleList(
            [
                TransformerEncoder(
                    hidden_size=model_config.t_hidden_size,
                    num_heads=model_config.t_n_head,
                    dropout_p=model_config.t_dropout_p,
                )
                for _ in range(model_config.t_TF_encoder_num)
            ]
        )
    def forward(self, x, pos):
        x = self.text_emb_layer(x)
        pos_emb = self.pos_emb_layer(pos)
        pos_mask = pos.lt(1)
        
        for conv_block in self.conv_blocks:
            x = conv_block(x, pos_mask)
        
        x = x + pos_emb 
        
        for tf_block in self.TF_encoder:
            x, _ = tf_block(x, pos_mask)
        
        return x


class ResidualEncoder(nn.Module):
    def __init__(self, model_config):
        super(ResidualEncoder, self).__init__()

        self.pos_emb_layer = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                model_config.max_seq_len+1,
                model_config.r_hidden_size[0],
                padding_idx=0
            ),
            freeze=True
        )
        self.pos_emb_layer.requires_grad_ = False
        
        self.lconv_blocks = nn.ModuleList(
            [
                LConvBlock(
                    hidden_size=model_config.r_hidden_size[0],
                    kernel_size=model_config.r_lconv_kernel_size,
                    num_heads=model_config.r_n_head,
                    dropout=model_config.r_dropout_p,
                    stride=1
                )
                for _ in range(model_config.r_lconv_num)
            ]
        )

        self.lconv_blocks_2 = nn.ModuleList(
            [
                LConvBlock(
                    hidden_size=model_config.r_hidden_size[0] if i == 0 else model_config.r_hidden_size[i-1],
                    kernel_size=model_config.r_lconv_kernel_size,
                    num_heads=model_config.r_n_head,
                    dropout=model_config.r_dropout_p,
                    stride=1
                )
                for i in range(model_config.r_lconv_with_conv)
            ]
        )
        self.convs = nn.ModuleList(
            [
                ConvNorm1D(
                    in_channels=model_config.r_hidden_size[0] if i == 0 else model_config.r_hidden_size[i-1],
                    out_channels=model_config.r_hidden_size[i],
                    kernel_size=model_config.r_conv_kernel_size,
                    stride=2,
                    padding=1
                )
                for i in range(model_config.r_lconv_with_conv)
            ]
        )
        self.pool_layer = nn.AdaptiveAvgPool2d((1, model_config.r_pool_size))
        
        self.projection = LinearNorm(
            model_config.r_pool_size, 
            model_config.r_pool_size*4
        )

        self.r_lconv_with_conv = model_config.r_lconv_with_conv
        
    def forward(self, x, pos):

        pos_mask = pos.lt(1)

        for lconv_block in self.lconv_blocks:
            x = lconv_block(x, pos_mask)

        for i in range(self.r_lconv_with_conv):
            x = self.lconv_blocks_2[i](x, pos_mask if i == 0 else None)
            x = self.convs[i](x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)

        x = self.projection(self.pool_layer(x).squeeze())
        
        return x
