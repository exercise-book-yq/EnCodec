from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from utils import AttrDict
from utils import mel_spectrogram, MAX_WAV_VALUE, load_wav, get_dataset_filelist
from model import EncodecModel
from data import SoundDataset, get_dataloader
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
import hydra
h = None
device = None


# import Time

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(config, a):
    model = EncodecModel.my_encodec_model(a.checkpoint_file).to(device)



    # training_filelist, validation_filelist = get_dataset_filelist(a)

    # validset = SoundDataset(
    #         validation_filelist,
    #         split=False,
    #         shuffle=False,
    #         segment_size=h.segment_size,
    #         target_sample_hz=soundstream.target_sample_hz,
    #         seq_len_multiple_of=soundstream.seq_len_multiple_of
    #     )
    # validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
    #                                    sampler=None,
    #                                    batch_size=1,
    #                                    pin_memory=True,
    #                                    drop_last=True)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    # state_dict_do = load_checkpoint(cp_do, device)
    # model.load_state_dict(state_dict_g['generator'])

    model.set_target_bandwidth(1.5)
    # mpd.load_state_dict(state_dict_do['mpd'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()
    for filepath in tqdm(os.listdir(a.input_wavs_dir)):
        filelist = os.listdir(a.input_wavs_dir + filepath)
        filelist.sort()
        # print(filepath)
        # filename = filelist

        if not os.path.exists(a.output_dir + filepath + '/'):
            os.makedirs(a.output_dir + filepath + '/')

        with torch.no_grad():
            for i, filename in enumerate(filelist):

                wave, sr = load_wav(os.path.join(a.input_wavs_dir + filepath, filename))
                wave = wave / MAX_WAV_VALUE
                # print(wave)
                wave = torch.FloatTensor(wave)

                wave = wave.to(device)
                # x = torch.transpose(x, 1, 2)
                # x = torch.DoubleTensor(x).to(device)
                # a = Time.time()
                wave = wave.unsqueeze(0)
                wave = wave.unsqueeze(0)
                y_g_hat = model(wave)
                # b =  Time.time()
                # print(b-a)
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')

                output_file = os.path.join(a.output_dir + filepath + '/', os.path.splitext(filename)[0] + '.wav')
                write(output_file, config.model.sample_rate, audio)
                # print(output_file)

@hydra.main(config_path='config', config_name='config-2')
def main(config):
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default="/home/Datasets1/youqiang/VCTK/Stream/Inter-test/target/16k-1/")
    parser.add_argument('--output_dir',
                        default='/home/Datasets1/youqiang/encodec-results/test/1500bps/v7-1-libri/')
    parser.add_argument('--checkpoint_file',
                        default='/home/Datasets1/youqiang/cp_hifigan/Librispeech/encodec/v7-1/g_00300000')
    parser.add_argument('--input_training_file',
                        default='/home/youqiang/code/SoundStream/audiolm-pytorch-hifigan/VCTK/training.txt')
    parser.add_argument('--input_validation_file',
                        default='/home/youqiang/code/SoundStream/audiolm-pytorch-hifigan/VCTK/validation.txt')
    a = parser.parse_args()

    # config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    # with open(config_file) as f:
    #     data = f.read()
    #
    # global h
    # json_config = json.loads(data)
    # h = AttrDict(json_config)
    #
    # torch.manual_seed(h.seed)
    # global device
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(h.seed)
    #     device = torch.device('cuda')
    # else:
    global device
    device = torch.device('cuda')

    inference(config, a)


if __name__ == '__main__':
    main()

