from .blocks import ConvBlock, TransformerEncoder, LConvBlock, ConvNorm1D, LinearNorm, SwishBlock
import torch
from torch import nn
from utils import get_sinusoid_encoding_table, get_mask_from_lengths
import numpy as np
class TextEncdoer(nn.Module):
    def __init__(self, model_config, vocab_size, activation=nn.ReLU()):
        super(TextEncdoer, self).__init__()
        
        self.text_emb_layer = nn.Embedding(vocab_size ,model_config.t_embedding_size)
        
        self.pos_emb_layer = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                model_config.max_seq_len+1,
                model_config.hidden_size,
                padding_idx=0
            ),
            freeze=True
        )
        self.pos_emb_layer.requires_grad_ = False
        
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channel=model_config.hidden_size, 
                    out_channel=model_config.hidden_size, 
                    dropout=model_config.t_dropout_p, 
                    activation=activation
                )
                for _ in range(model_config.t_conv_num)
            ]
        )

        self.TF_encoder = nn.ModuleList(
            [
                TransformerEncoder(
                    hidden_size=model_config.hidden_size,
                    num_heads=model_config.n_head,
                    dropout_p=model_config.t_dropout_p,
                )
                for _ in range(model_config.t_TF_encoder_num)
            ]
        )
    def forward(self, x, pos_text, text_mask):
        x = self.text_emb_layer(x)
        pos_emb = self.pos_emb_layer(pos_text)
                
        for conv_block in self.conv_blocks:
            x = conv_block(x, text_mask)
        
        x = x + pos_emb 
        
        for tf_block in self.TF_encoder:
            x, _ = tf_block(x, text_mask)
        
        return x


class ResidualEncoder(nn.Module):
    def __init__(self, model_config, num_mel=80):
        super(ResidualEncoder, self).__init__()

        self.latent_size = model_config.r_latent_size
                
        self.pos_emb_layer = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                model_config.max_seq_len+1,
                num_mel,
                padding_idx=0
            ),
            freeze=True
        )
        self.pos_emb_layer.requires_grad_ = False
        
        self.lconv_blocks = nn.ModuleList(
            [
                LConvBlock(
                    hidden_size=model_config.hidden_size,
                    kernel_size=model_config.r_lconv_kernel_size,
                    num_heads=model_config.n_head,
                    dropout=model_config.r_dropout_p,
                    stride=1
                )
                for _ in range(model_config.r_lconv_num)
            ]
        )
        
        self.input_linear = LinearNorm(
            num_mel*2+model_config.speaker_emb, model_config.hidden_size
        )
        
        self.text_encoder_norm = nn.LayerNorm(model_config.hidden_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=model_config.hidden_size, 
            num_heads=1, 
            batch_first=True
        )

        self.fc_mu = LinearNorm(model_config.hidden_size, model_config.r_latent_size)
        self.fc_var = LinearNorm(model_config.hidden_size, model_config.r_latent_size)
        
        self.out_linear = nn.Sequential(
            LinearNorm(
            self.latent_size + model_config.speaker_emb + model_config.hidden_size,
            model_config.r_out_size
            ),
            nn.Tanh()
        )
        self.out_layer_norm = nn.LayerNorm(model_config.r_out_size)
        
    def reparameterize(self, mu, logvar):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, text_encoder_out, pos_text, text_mask, mel, pos_mel, mel_mask, speaker_emb, max_src_len=1000):

        text_encoding = self.text_encoder_norm(text_encoder_out)
        bs = text_encoding.size(0)
        
        if not self.training and mel is None:
            mu = log_var = torch.zeros([bs, text_mask.size(1), self.latent_size])
            attn = None
        else:
            mel_pos_emb = self.pos_emb_layer(pos_mel)
            speaker_embedding_m = speaker_emb.unsqueeze(1).expand(-1, mel_pos_emb.size(1), -1)
            
            x = torch.cat([mel, mel_pos_emb, speaker_embedding_m], dim=-1)
            x = self.input_linear(x)
            for lconv_block in self.lconv_blocks:
                x = lconv_block(x, mel_mask)

            x, attn = self.attention(
                query=text_encoding, 
                key=x, 
                value=x, 
                key_padding_mask=mel_mask)
            
            mu = self.fc_mu(x).masked_fill(text_mask.unsqueeze(-1), 0)
            log_var = self.fc_var(x).masked_fill(text_mask.unsqueeze(-1), 0)
        
        # Phoneme-Level Fine-Grained VAE
        z = self.reparameterize(mu, log_var)
        z = z.masked_fill(text_mask.unsqueeze(-1), 0)
        
        speaker_embedding_t = speaker_emb.unsqueeze(
            1).expand(-1, text_mask.size(1), -1)
        
        x = torch.cat([z, speaker_embedding_t, text_encoding], dim=-1)
        x = self.out_layer_norm(self.out_linear(x))
        x = x.masked_fill(text_mask.unsqueeze(-1), 0)
        
        return x, attn, mu, log_var
    

class DurationPredictor(nn.Module):
    def __init__(self, model_config):
        super(DurationPredictor, self).__init__()
        
        self.lconv_blocks = nn.ModuleList(
            [
                LConvBlock(
                    hidden_size=model_config.r_out_size,
                    kernel_size=model_config.d_lconv_kernel_size,
                    num_heads=model_config.n_head,
                    dropout=model_config.d_dropout_p,
                    stride=1
                )
                for _ in range(model_config.d_lconv_num)
            ]
        )
        
        self.projection = nn.Sequential(
            LinearNorm(model_config.r_out_size, 1),
            nn.Softplus()
        )
    def forward(self, x, text_mask=None):
        
        for block in self.lconv_blocks:
            x = block(x, text_mask)

        dur = self.projection(x)
        if text_mask is not None:
            dur = dur.masked_fill(text_mask.unsqueeze(-1), 0)
        
        return x, dur.squeeze()
    
class LearnedUpsampling(nn.Module):
    def __init__(self, model_config):
        super(LearnedUpsampling, self).__init__()
        self.conv_c = ConvBlock(
            model_config.r_out_size, 
            model_config.l_conv_out_size,
            0, 
            model_config.l_conv_kernel_size,
            nn.SiLU()
        )
        
        self.swish_c = SwishBlock(
            model_config.l_conv_out_size+2,
            model_config.l_swish_c_out_size,
            model_config.l_swish_c_out_size,
        )

        self.conv_w = ConvBlock(
            model_config.r_out_size, 
            model_config.l_conv_out_size,
            0, 
            model_config.l_conv_kernel_size,
            nn.SiLU()
        )
        
        self.swish_w = SwishBlock(
            model_config.l_conv_out_size+2,
            model_config.l_swish_w_out_size,
            model_config.l_swish_w_out_size,
        )
        
        self.proj_w = LinearNorm(model_config.l_swish_w_out_size, 1)
        self.softmax_w = nn.Softmax(dim=2)

        self.linear_einsum = LinearNorm(model_config.l_swish_c_out_size, model_config.r_out_size) # A
        self.layer_norm = nn.LayerNorm(model_config.r_out_size)
    
    def forward(self, dur, V, pos_text, text_mask):
        
        # Duration Interpretation
        batch_size = text_mask.size(0)
        pred_mel_len = torch.round(dur.sum(-1)).type_as(pos_text)
        pred_mel_len = torch.clamp(pred_mel_len, max=1000)
        pred_max_mel_len = pred_mel_len.max().item()
        pred_mel_mask = get_mask_from_lengths(pred_mel_len, pred_max_mel_len)
        
        # Prepare Attention Mask
        text_mask_ = text_mask.unsqueeze(
            1).expand(-1, pred_mel_mask.shape[1], -1)  # [B, tat_len, src_len]
        pred_mel_mask_ = pred_mel_mask.unsqueeze(-1).expand(-1, -1, text_mask.shape[1])
        attn_mask = torch.zeros((text_mask.shape[0], pred_mel_mask.shape[1], text_mask.shape[1])).type_as(pos_text)
        
        attn_mask = attn_mask.masked_fill(text_mask_, 1.)
        attn_mask = attn_mask.masked_fill(pred_mel_mask_, 1.)
        attn_mask = attn_mask.bool()
        
        # Token Boundary Grid
        e_k = torch.cumsum(dur, dim=1)
        s_k = e_k - dur
        e_k = e_k.unsqueeze(1).expand(batch_size, pred_max_mel_len, -1)
        s_k = s_k.unsqueeze(1).expand(batch_size, pred_max_mel_len, -1)
        
        t_arange = torch.arange(1, pred_max_mel_len+1).type_as(pos_text).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, attn_mask.size(2)
        )
        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(attn_mask, 0)
        
        # Attention (W)
        W = self.swish_w(S, E, self.conv_w(V)) # [B, T, K, dim_w]
        W = self.proj_w(W).squeeze(-1).masked_fill(text_mask_, -np.inf) #[B, T, K]
        W = self.softmax_w(W) #[B, T, K]
        W = W.masked_fill(pred_mel_mask_, 0.)
        
        # Auxiliary Attention Context (C)
        C = self.swish_c(S, E, self.conv_c(V))

        # Out
        upsampled_rep = torch.matmul(W, V) + self.linear_einsum(torch.einsum('btk,btkp->btp', W, C)) # [B, T, M]
        upsampled_rep = self.layer_norm(upsampled_rep)
        upsampled_rep = upsampled_rep.masked_fill(pred_mel_mask.unsqueeze(-1), 0)

        return upsampled_rep, pred_mel_mask, pred_mel_len, W

class Decoder(nn.Module):
    def __init__(self, model_config, num_mel=80):
        super(Decoder, self).__init__()
        
        self.max_seq_len = model_config.max_seq_len
        self.n_layers = model_config.dc_layer_num
        self.pos_emb_layer = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                model_config.max_seq_len+1,
                model_config.r_out_size,
                padding_idx=0
            ),
            freeze=True
        )
        self.pos_emb_layer.requires_grad_ = False

        self.lconv_blocks = nn.ModuleList(
            [
                LConvBlock(
                    hidden_size=model_config.r_out_size,
                    kernel_size=model_config.dc_lconv_kernel_size,
                    num_heads=model_config.n_head,
                    dropout=model_config.dc_dropout_p,
                    stride=1
                )
                for _ in range(model_config.dc_layer_num)
            ]
        )
        self.projection = nn.ModuleList(
            [
                LinearNorm(
                    model_config.r_out_size, num_mel
                )
                for _ in range(model_config.dc_layer_num)
            ]
        )
        
    def forward(self, x, mask):
        mel_iters = []
        batch_size, max_len = x.shape[0], x.shape[1]

        if not self.training and max_len > self.max_seq_len:
            pos_emb = self.pos_emb_layer(torch.arange(max_len).cuda()).unsqueeze(
                0).expand(batch_size, -1, -1)
            x = x + pos_emb
        else:
            max_len = min(max_len, self.max_seq_len)
            pos_emb = self.pos_emb_layer(torch.arange(max_len).cuda()).unsqueeze(
                0).expand(batch_size, -1, -1)
            mask = mask[:, :max_len]

        for i, (conv, linear) in enumerate(zip(self.lconv_blocks, self.projection)):
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = torch.tanh(conv(
                x, mask=mask
            ))
            if self.training or not self.training and i == self.n_layers-1:
                mel_iters.append(
                    linear(x).masked_fill(mask.unsqueeze(-1), 0)
                )
        return mel_iters, mask