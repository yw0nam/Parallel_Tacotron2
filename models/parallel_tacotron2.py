import torch.nn as nn
import torch
from models.modules import ResidualEncoder, TextEncdoer, DurationPredictor, LearnedUpsampling, Decoder

class ParallelTacotron2(nn.Module):
    def __init__(self, model_config, vocab_len, num_mels, num_speakers):
        super(ParallelTacotron2, self).__init__()
        
        self.vocab_len = vocab_len
        self.num_mels = num_mels
        self.num_speakers = num_speakers
        
        self.text_encoder = TextEncdoer(model_config, self.vocab_len)
        self.residual_encoder = ResidualEncoder(model_config, self.num_mels)
        self.learned_upsampling = LearnedUpsampling(model_config)
        self.duration_predictor = DurationPredictor(model_config)
        self.decoder = Decoder(model_config)
        self.speaker_emb_layer = nn.Embedding(num_speakers, model_config.speaker_emb, 0)
        
    def forward(self, text, mel, pos_text, pos_mel, speaker_id):
        
        speaker_emb = self.speaker_emb_layer(speaker_id)
        
        text_mask = pos_text.lt(1)
        mel_mask = pos_mel.lt(1)
        
        text_out = self.text_encoder(text, text_mask)
        
        x, attn, mu, log_var = self.residual_encoder(
            text_out, text_mask, mel, 
            mel_mask, speaker_emb
        )
        
        v, dur = self.duration_predictor(x, text_mask)

        upsampled_rep, pred_mel_mask, _, W = self.learned_upsampling(dur, 
                                                                    v, 
                                                                    pos_text, 
                                                                    text_mask
                                                                    )
        
        mel_iters, _ = self.decoder(upsampled_rep, pred_mel_mask)
        
        
        return {
            'mel_iters' : mel_iters,
            'dur': dur,
            'mu': mu,
            'log_var': log_var,
            'attn' : attn,
            'W' : W
        }
