import torch
from torch import nn
import pytorch_lightning as pl
import transformers
from models.parallel_tacotron2 import ParallelTacotron2
from models.loss import ParallelTacotron2Loss

class PL_model(pl.LightningModule):
    def __init__(self, train_config, model_config, vocab_size, num_mels, num_speakers):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_config = train_config
        self.model_config = model_config
        self.vocab_size= vocab_size
        self.num_mels = num_mels,
        self.num_speakers = num_speakers
        
        self.model = ParallelTacotron2(model_config, vocab_size, num_mels, num_speakers)
        self.loss = ParallelTacotron2Loss(train_config)
        
    def forward(self, data):
        return self.model(**data)
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        preds = self.forward(data)
        
        total_loss, mel_loss, duration_loss, kl_loss, beta = self.loss(preds, labels,self.global_step).values()
        
        self.log("total_loss",  total_loss, on_epoch=False, on_step=True)
        self.log("mel_loss", mel_loss, on_epoch=False, on_step=True)
        self.log("duration_loss", duration_loss, on_epoch=False, on_step=True)
        self.log("kl_loss", kl_loss, on_epoch=False, on_step=True)
        self.log("beta", beta, on_epoch=False, on_step=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        preds = self.forward(data)

        total_loss, mel_loss, duration_loss, kl_loss, _ = self.loss(
            preds, labels, self.global_step).values()

        self.log("val_total_loss",  total_loss, on_epoch=True, on_step=False)
        self.log("val_mel_loss", mel_loss, on_epoch=True, on_step=False)
        self.log("val_duration_loss", duration_loss, on_epoch=True, on_step=False)
        self.log("val_kl_loss", kl_loss, on_epoch=True, on_step=False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.train_config.lr)

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config.warmup_step,
            num_training_steps=self.train_config.training_step
        )
        return [optimizer], [scheduler]
