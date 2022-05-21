from configs import *
import os
from models.trainer import fit_model

save_path = '/data1/spow12/model_weights/TTS/parallel_tacotron2/'

data_config = DataConfig(
    root_dir='/data1/spow12/datas/TTS/LJSpeech-1.1'
)
train_config = TrainConfig(
    batch_size=16,
    checkpoint_path=os.path.join(save_path, 'checkpoint'),
    log_dir=os.path.join(save_path, 'tensorboard')
)
model_config = ModelConfig()

fit_model(model_config, data_config, train_config)