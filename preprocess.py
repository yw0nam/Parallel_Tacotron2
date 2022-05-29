import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', required=True, type=str)
    p.add_argument('--clip_length_lower', default=40, type=int)
    p.add_argument('--clip_length_upper', default=200, type=int)
    config = p.parse_args()

    return config   

def main(args):
    csv = pd.read_csv(args.csv_path, sep='|', names=['wav_name', 'ori_text', 'norm_text'])
    csv['speaker_idx'] = 0
    csv = csv[~csv['norm_text'].isna()]
    csv['length'] = csv['norm_text'].map(lambda x: len(x))
    
    csv = csv.query('length >= %d and length <= %d' %
                    (args.clip_length_lower, args.clip_length_upper))
    dev, test = train_test_split(csv, test_size=100, random_state=1004)
    train, val = train_test_split(dev, test_size=500, random_state=1004)
    
    save_path = os.path.dirname(args.csv_path)
    
    train = train.sort_values(by=['length'], ascending=False).drop('length', axis=1)
    val = val.sort_values(by=['length'], ascending=False).drop('length', axis=1)
    test = test.sort_values(by=['length'], ascending=False).drop('length', axis=1)
    
    train.to_csv(os.path.join(save_path, "metadata_train.csv"),index=False, header=None, sep='|')
    val.to_csv(os.path.join(save_path, "metadata_val.csv"),index=False, header=None, sep='|')
    test.to_csv(os.path.join(save_path, "metadata_test.csv"),index=False, header=None, sep='|')
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)