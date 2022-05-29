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
        plt.rcParams['font.size'] = '16'
        if self.draw_step <= self.global_step:
            for out in validation_step_outputs:
                Ws = out['W'].detach().cpu()
                fig = plt.figure(figsize=(20, 10))
                for i in range(1, 3):
                    xlims = Ws[i-1].shape[1]
                    ax = fig.add_subplot(2, 1, i)
                    im = ax.imshow(Ws[i-1], origin='lower', aspect='auto')
                    ax.set_xlabel('Decoder timestep')
                    ax.set_ylabel('Encoder timestep')
                    ax.set_xlim(0, xlims)
                    ax.tick_params(labelsize="x-small")
                    ax.set_anchor("W")
                    fig.colorbar(im)
                # self.writer.add_figure('attention', fig, self.global_step)
                self.logger.experiment.add_figure(
                    'log_%d'%(self.global_step), fig, self.global_step)
                self.draw_step += self.train_config.attn_draw_step
                plt.close('all')
                break
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_config.lr)
        
        scheduler = ScheduledOptim(optimizer, self.train_config)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
