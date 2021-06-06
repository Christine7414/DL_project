from torch.optim import Adam
import torch
import numpy as np
from generate import sample_batch, conv_params, gru_params
from model import ARC, Enhancement
from scipy import signal
import torch.nn as nn
import musdb

def gru_params(model) :
    for name, params in model.named_parameters() :
        if(name[:3] == 'gru') :
            yield params
def conv_params(model) :
    for name, params in model.named_parameters() :
        if(name[:3] != 'gru') :
            yield params


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use {} on this device'.format(device))
epochs = 1
mus = musdb.DB('./musdb18', subsets = 'train')
separation_model = ARC().to(device)
criterion = nn.MSELoss()
gru_optim = Adam(gru_params(separation_model), lr = 1e-4)
conv_optim = Adam(conv_params(separation_model), lr = 1e-3)
batch_size = 10
mus = musdb.DB('./musdb18')
if __name__ == '__main__' :
    for epoch in range(epochs) :
        for iter in range(100) :
            raw = sample_batch(mus, batch_size)
            _, _, mix_stft = signal.stft(raw['mix'], nperseg=2048, noverlap=1024)
            _, _, drum_stft = signal.stft(raw['drum'], nperseg=2048, noverlap=1024)
            _, _, bass_stft = signal.stft(raw['bass'], nperseg=2048, noverlap=1024)
            _, _, other_stft = signal.stft(raw['other'], nperseg=2048, noverlap=1024)
            _, _, vocal_stft = signal.stft(raw['vocal'], nperseg=2048, noverlap=1024)
            x = np.concatenate((mix_stft[:, 0, :, :], mix_stft[:, 1, :, :]), axis = 0)
            target_1 = np.concatenate((drum_stft[:, 0, :, :], bass_stft[:, 0, :, :], other_stft[:, 0, :, :], vocal_stft[:, 0, :, :]), axis = 1)
            target_2 = np.concatenate((drum_stft[:, 1, :, :], bass_stft[:, 1, :, :], other_stft[:, 1, :, :], vocal_stft[:, 1, :, :]), axis = 1)
            target = np.concatenate((target_1, target_2), axis = 0)
            
            x = np.log10(np.abs(x) ** 2 + 1)
            x = torch.FloatTensor(x).to(device)

            target = np.log10(np.abs(target) ** 2 + 1)
            target = torch.FloatTensor(target).to(device)
            y_hat = separation_model(x)
            loss = criterion(y_hat, target)
            gru_optim.zero_grad()
            conv_optim.zero_grad()
            loss.backward()
            gru_optim.step()
            conv_optim.step()
    print("Epoch : {} Loss : {}".format(epoch + 1, loss.item()))



