import os
import pandas as pd
import pickle
import torch
import torchaudio
import torchaudio.transforms as T

file_path = 'data_2020/'
csv_file = 'data_2020/evaluation_setup/fold1_all.csv' # train + evaluate
output_path = 'features/logmel128_scaled_d_dd'
feature_type = 'logmel'

sr = 48000
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)

def cal_deltas(x):
    out = (x[:, :, 2:] - x[:, :, :-2]) / 10.0
    out = out[:, :, 1:-1] + (x[:, :, 4:] - x[:,:,:-4]) / 5.0
    return out

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()

mel_spec = T.MelSpectrogram(sr, num_fft, hop_length=hop_length, n_mels=num_freq_bin, f_min=0.0, f_max=sr/2, mel_scale='htk')

for i in range(len(wavpath)):
    wave, fs = torchaudio.load(file_path + wavpath[i])
    logmel = torch.log(mel_spec(wave) + 1e-8)
    feat = (logmel - torch.min(logmel)) / (torch.max(logmel) - torch.min(logmel))
    deltas = cal_deltas(feat)
    deltas_deltas = cal_deltas(deltas)

    feat = torch.cat((feat[:, :, 4:-4], deltas[:, :, 2:-2], deltas_deltas))
    feature = {'feat': feat,}
    cur_file_name = output_path + wavpath[i][5:-3] + feature_type
    pickle.dump(feature, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
