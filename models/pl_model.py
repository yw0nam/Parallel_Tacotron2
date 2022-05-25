import torch
from torch import nn
import pytorch_lightning as pl
from models.scheduler import ScheduledOptim
from models.parallel_tacotron2 import ParallelTacotron2
from models.loss import ParallelTacotron2Loss
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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
        self.draw_step = 0
        
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
        
        return preds
    
    def validation_epoch_end(self, validation_step_outputs):
        if self.draw_step >= self.global_step:
            for out in validation_step_outputs:
                attn = out['attn'].detach().cpu()
                fig = plt.figure(figsize=(25, 15))
                for i in range(1, self.train_config.batch_size + 1):
                    ax = fig.add_subplot(self.train_config.batch_size, 1, i)
                    ax.imshow(attn[i-1])
                    ax.set_title("attention_%d" % i)
                # self.writer.add_figure('attention', fig, self.global_step)
                self.logger.experiment.add_figure(
                    'attention_%d'%(self.global_step), fig, self.global_step)
                self.draw_step += self.train_config.attn_draw_step
                plt.close('all')
                break
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_config.lr)
        
        scheduler = ScheduledOptim(optimizer, self.train_config)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
