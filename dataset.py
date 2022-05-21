import pandas as pd
from torch.utils.data import Dataset, DistributedSampler
import os
import librosa
import numpy as np
from text import phoneme_to_sequence, pad_with_eos_bos, text_to_sequence
import torch
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import torchaudio
from utils import _pad_data, _pad_mel, _prepare_data

class TTSdataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, config, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
        """
        if train:
            self.landmarks_frame = pd.read_csv(os.path.join(config.root_dir, config.train_csv), 
                                            sep='|', names=['wav_name', 'text_1', 'text_2', 'speaker_id'])
        else:
            self.landmarks_frame = pd.read_csv(os.path.join(config.root_dir, config.val_csv),
                                               sep='|', names=['wav_name', 'text_1', 'text_2', 'speaker_id'])
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=config.power,
            normalized=False
        )
        try:
            os.mkdir(self.config.phoneme_cache_path)
        except:
            pass
        
    def __len__(self):
        return len(self.landmarks_frame)
    
    def _load_data(self, idx):
        wav_name = self.landmarks_frame['wav_name'].iloc[idx]
        wav_path = os.path.join(self.config.root_dir,
                                'wavs', wav_name) + '.wav'
        text = self.landmarks_frame['text_1'].iloc[idx]
        cache_path = os.path.join(
            self.config.phoneme_cache_path, wav_name+'.npy')

        if self.config.use_phonemes:
            seq = TTSdataset._load_or_generate_phoneme_sequence(text, wav_name, cache_path,
                                                                self.config.cleaners,
                                                                self.config.language,
                                                                self.config.enable_eos_bos)
        else:
            seq = np.asarray(
                text_to_sequence(
                    text,
                    self.config.cleaners,
                ),
                dtype=np.int32,
            )

        mel = self.load_wav(wav_path)
        pos_text = np.arange(1, len(seq) + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)
        speaker_id = self.landmarks_frame['speaker_id'].iloc[idx]
        
        sample = {
            'text': seq,
            'mel': mel,
            'pos_text': pos_text,
            'pos_mel': pos_mel,
            'speaker_id': speaker_id
        }

        return sample

    def __getitem__(self, idx):
        sample = self._load_data(idx)
        return sample

    def load_wav(self, filename):
        wav, _ = torchaudio.load(filename)
        margin = int(self.config.sr * 0.01)
        wav = librosa.effects.trim(
            wav[:, margin:-margin], top_db=self.config.trim_db, frame_length=self.config.win_length, hop_length=self.config.hop_length)[0]
        mel = self.mel_transform(wav)
        mel_db = 20 * np.log10(np.maximum(1e-5,  mel))
        mel_norm = (mel_db - self.config.ref_db -
                    self.config.min_level_db) / (-self.config.min_level_db)
        mel_norm = ((2 * self.config.max_norm) * mel_norm) - self.config.max_norm
        mel_norm_cliped = np.clip(mel_norm, -self.config.max_norm, self.config.max_norm)
        return mel_norm_cliped.squeeze().transpose(1, 0)
    
    @staticmethod
    def _generate_and_cache_phoneme_sequence(
        text, cache_path, cleaners, language, custom_symbols=None, characters=None, add_blank=None
    ):
        """generate a phoneme sequence from text.
            since the usage is for subsequent caching, we never add bos and
            eos chars here. Instead we add those dynamically later; based on the
            config option."""
        phonemes = phoneme_to_sequence(
            text,
            [cleaners],
            language=language,
            enable_eos_bos=False,
            custom_symbols=custom_symbols,
            tp=characters,
            add_blank=add_blank,
        )
        phonemes = np.asarray(phonemes, dtype=np.int32)
        np.save(cache_path, phonemes)
        return phonemes
    
    @staticmethod
    def _load_or_generate_phoneme_sequence(text, wav_name, cache_path, cleaners, language, enable_eos_bos=True):
        try:
            seq = np.load(cache_path)
        except FileNotFoundError:
            seq = TTSdataset._generate_and_cache_phoneme_sequence(text, cache_path, cleaners, language)
        except (ValueError, IOError):
            print(
                " [!] failed loading phonemes for {}. " "Recomputing.".format(wav_name))
            seq = TTSdataset._generate_and_cache_phoneme_sequence(text, cache_path, cleaners, language)
        if enable_eos_bos:
            seq = pad_with_eos_bos(seq)
            seq = np.asarray(seq, dtype=np.int32)
        return seq



class Transformer_Collator():

    def __init__(self):
        pass 
    
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        speaker_id = [d['speaker_id'] for d in batch]
        text_length = [len(d['text']) for d in batch]
        mel_length = [len(d['mel']) for d in batch]
        
        
        text = [i for i, _ in sorted(
            zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(
            zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(
            zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(
            zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_length = [i for i, _ in sorted(
            zip(mel_length, text_length), key=lambda x: x[1], reverse=True)]
        text_length = [i for i in sorted(
            zip(text_length), key=lambda x: x[0], reverse=True)]        
        
        text =_prepare_data(text).astype(np.int32)
        mel = _pad_mel(mel)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text =_prepare_data(pos_text).astype(np.int32)
        model_input = {
            'text': torch.LongTensor(text),
            'mel' : torch.FloatTensor(mel),
            'pos_text': torch.LongTensor(pos_text),
            'pos_mel': torch.LongTensor(pos_mel),
            'speaker_id':torch.LongTensor(speaker_id),
        }
        label = {
            'mel': torch.FloatTensor(mel),
            'mel_length': torch.LongTensor(mel_length),
            'text_length': torch.LongTensor(text_length)
        }
        return model_input, label
    
class PartitionPerEpochDataModule(pl.LightningDataModule):

    def __init__(
        self, batch_size, config, num_workers=4
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def prepare_data(self):
        pass
    def setup(self, stage: Optional[str] = None):
        """
        Anything called here is being distributed across GPUs
        (do many times).  Lightning handles distributed sampling.
        """
        # Build the val dataset
        
        self.val_dataset = TTSdataset(self.config,  train=False)
        self.train_dataset = TTSdataset(self.config,  train=True)
        
    def train_dataloader(self):
        """
        This function sends the same file to each GPU and
        loops back after running out of files.
        Lightning will apply distributed sampling to
        the data loader so that each GPU receives
        different samples from the file until exhausted.
        """
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Transformer_Collator(self.preprocessor),
            pin_memory=True,
            shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Transformer_Collator(self.preprocessor),
            pin_memory=True,
        )
