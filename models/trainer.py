import torch, os
from models.pl_model import PL_model
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from dataset import Transformer_Collator, TTSdataset
from pytorch_lightning.callbacks import LearningRateMonitor

def fit_model(model_config, data_config, train_config):
    
    pl.seed_everything(42)
    num_gpu = torch.cuda.device_count()
    
    train_dataset = TTSdataset(data_config, train=True)
    val_dataset = TTSdataset(data_config, train=False)
    
    train_loader = DataLoader(train_dataset, train_config.batch_size, num_workers=train_config.num_workers,
                            collate_fn=Transformer_Collator(),
                            pin_memory=True, shuffle=False)
    
    val_loader = DataLoader(val_dataset, train_config.batch_size, 
                            num_workers=train_config.num_workers,
                            collate_fn=Transformer_Collator(), 
                            pin_memory=True,shuffle=False)
    
    model = PL_model(
        train_config, 
        model_config,
        data_config.vocab_size,
        data_config.n_mels,
        data_config.speaker_num
    )
    setattr(model, 'train_dataloader', lambda: train_loader)
    setattr(model, 'val_dataloader', lambda: val_loader)
    
    checkpoint_callback = plc.ModelCheckpoint(
        monitor="step",
        dirpath=os.path.join(train_config.checkpoint_path, train_config.exp_name),
        filename="{step:06d}",
        save_top_k=5,
        mode="max",
    )
    
    logger = TensorBoardLogger(train_config.log_dir, name=train_config.exp_name, version=train_config.log_version)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpu,
        strategy="ddp",
        max_steps=train_config.training_step,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,
        amp_backend="native",
        profiler="simple",
        accumulate_grad_batches=train_config.accumulate_grad,
        logger=logger,
        gradient_clip_val=train_config.gradient_clip,
    )
    trainer.fit(model, ckpt_path=train_config.resume_from_checkpoint)
