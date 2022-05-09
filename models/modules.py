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
    def forward(self, x, pos_text):
        x = self.text_emb_layer(x)
        pos_emb = self.pos_emb_layer(pos_text)
        pos_mask = pos_text.lt(1)
                
        for conv_block in self.conv_blocks:
            x = conv_block(x, pos_mask)
        
        x = x + pos_emb 
        
        for tf_block in self.TF_encoder:
            x, _ = tf_block(x, pos_mask)
        
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
    
    def forward(self, text_encoder_out, pos_text, mel, pos_mel, speaker_emb, max_src_len=1000):

        text_encoding = self.text_encoder_norm(text_encoder_out)
        bs = text_encoding.size(0)
        text_mask = pos_text.lt(1)
        
        if not self.training and mel is None:
            mu = log_var = torch.zeros([bs, text_mask.size(1), self.latent_size])
            attn = None
        else:
            mel_pos_emb = self.pos_emb_layer(pos_mel)
            mel_mask = pos_mel.lt(1)
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
    def forward(self, x, pos_text=None):
        
        text_mask = pos_text.lt(1)
        for block in self.lconv_blocks(x):
            x = block(x, text_mask)

        dur = self.projection(x)
        if text_mask is not None:
            dur = dur.masked_fill(text_mask.unsqueeze(-1), 0)
        
        return x, dur.squeeze()