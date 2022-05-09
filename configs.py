from dataclasses import dataclass, field

@dataclass
class DataConfig():
    """
    Data Settings
    """
    
    # Audio Config
    n_fft: int = 1024
    sr: int = 22050
    preemphasis: float = 0.97
    hop_length: int = 256  
    win_length: int = 1024
    n_mels: int  = 80  # Number of Mel banks to generate
    power: float = 1.5  # Exponent for amplifying the predicted magnitude
    min_level_db: int = -100
    ref_level_db: int = 20
    trim_db: int = 60
    ref_db: int = 20
    max_norm: int = 1
    # Text Config 
    cleaners: str = "phoneme_cleaners"
    use_phonemes: bool =True
    language: str ="en-us"
    phoneme_cache_path: str= './phonemes/'
    symbol_length: int = 134
    enable_eos_bos: bool = True
    # Data path config
    train_csv: str = 'metadata.csv'
    val_csv: str = 'metadata.csv'
    root_dir: str = './data/LJSpeech-1.1'
    train_samples: int = 12000
    
@dataclass
class TrainConfig():
    """
    Train Setting
    """
    bce_weight: int = 8
    hidden_size: int = 256
    decoder_prenet_hidden_size: int = 32
    n_head: int = 8
    embedding_size: int = 256
    n_layers: int = 4
    dropout_p: int = 0.2
    warmup_step: int = 1600
    training_step: int = 16000
    lr: float = 0.001
    batch_size: int = 128
    checkpoint_path: str = './models/checkpoint/'
    log_dir: str = './models/tensorboard/'
    

@dataclass
class ModelConfig():
    """
    Model Setting
    """
    # ================ #
    #  Global Setting  #
    # ================ #
    max_seq_len: int = 1000
    speaker_emb: int = 64
    # ================ #
    #   Text Encoder   #
    # ================ #
    t_hidden_size: int = 256
    t_n_head: int = 8
    t_embedding_size: int = 256
    t_dropout_p: float = 0.2
    t_conv_num: int = 3
    t_TF_encoder_num: int = 6

    # ================ #
    # Residual Encoder #
    # ================ #
    r_hidden_size: int = 256
    r_lconv_kernel_size: int = 17
    r_lconv_num: int = 5
    r_n_head: int = 8
    r_dropout_p: float = 0.2
    r_latent_size: int = 8
    r_out_size: int = 32