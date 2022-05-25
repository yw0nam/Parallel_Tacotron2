from dataclasses import dataclass, field

@dataclass
class DataConfig():
    """
    Data Settings
    """

    # ==================== #
    #     Audio Config     #
    # ==================== #
    n_fft: int = 1024
    sr: int = 22050
    hop_length: int = 256  
    win_length: int = 1024
    n_mels: int  = 80  # Number of Mel banks to generate
    power: float = 1.5  # Exponent for amplifying the predicted magnitude
    min_level_db: int = -100
    ref_level_db: int = 20
    trim_db: int = 60
    ref_db: int = 20
    max_norm: int = 1
    speaker_num: int = 1
    # ==================== #
    #     Text Config      #
    # ==================== #
    cleaners: str = "phoneme_cleaners"
    use_phonemes: bool = True
    language: str ="en-us"
    phoneme_cache_path: str= './phonemes/'
    vocab_size: int = 130
    enable_eos_bos: bool = True
    # ==================== #
    #   Data path config   #
    # ==================== #
    train_csv: str = 'metadata_train.csv'
    val_csv: str = 'metadata_val.csv'
    root_dir: str = './data/LJSpeech-1.1'
    
@dataclass
class TrainConfig():
    """
    Train Setting
    """
    # ==================== #
    #  Training arguments  #
    # ==================== #
    
    warmup_step: int = 1e+4
    training_step: int = 5e+5
    lr: float = 0.005
    batch_size: int = 4
    exp_name: str = './models'
    checkpoint_path: str = 'checkpoint/'
    log_dir: str = 'tensorboard/'
    log_version: str = None
    attn_draw_step: int = 1000
    num_workers: int = 8
    accumulate_grad: int = 16
    gradient_clip: float = 0.2
    anneal_steps: list = field(default_factory=lambda: [3e+5, 4e+5, 5e+5])
    anneal_rate: float = 0.3
    resume_from_checkpoint: str = None
    
    # ==================== #
    #     Loss Config      #
    # ==================== #
    gamma: float = 0.05
    warp: int = 128
    bandwidth: int = 60
    loss_lambda: int = 100 # For Duration loss
    kl_start: int = 6000
    kl_end: int = 50000
    kl_upper: float = 1
    
@dataclass
class ModelConfig():
    """
    Model Setting
    """
    # ================== #
    #   Global Setting   #
    # ================== #
    
    max_seq_len: int = 1000
    speaker_emb: int = 64
    hidden_size: int = 256
    n_head: int = 8
    
    # ================== #
    #    Text Encoder    #
    # ================== #
    
    t_embedding_size: int = 256
    t_dropout_p: float = 0.2
    t_conv_num: int = 3
    t_TF_encoder_num: int = 6

    # ================== #
    #  Residual Encoder  #
    # ================== #
    
    r_lconv_kernel_size: int = 17
    r_lconv_num: int = 5
    r_dropout_p: float = 0.2
    r_latent_size: int = 8
    r_out_size: int = 32
    
    # ================== #
    # Duration Predictor #
    # ================== #
    
    d_lconv_kernel_size: int = 5
    d_dropout_p: float = 0.2
    d_lconv_num: int = 4
    
    # ================== #
    # LearnedUpsampling  #
    # ================== #
    
    l_conv_out_size: int = 8
    l_conv_kernel_size: int = 3
    l_swish_c_out_size: int = 2
    l_swish_w_out_size: int = 16
    
    # ================== #
    #       Decoder      #
    # ================== #

    dc_lconv_kernel_size: int = 3
    dc_dropout_p: float = 0.1
    dc_layer_num: int = 6