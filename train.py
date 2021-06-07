from torch.optim import Adam
import torch
import numpy as np
from utils import conv_params, gru_params, sample_batch, transform
from model import ARC, Enhancement
import torch.nn as nn
import musdb

if __name__ == '__main__' :
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use {} on this device'.format(device))
    iterations = 10000
    mus_train = musdb.DB('./musdb18', subsets = 'train')
    mus_val = musdb.DB('./musdb18', subsets = 'test')
    separation_model = ARC().to(device)
    criterion = nn.MSELoss()
    gru_optim = Adam(gru_params(separation_model), lr = 1e-4)
    conv_optim = Adam(conv_params(separation_model), lr = 1e-3)
    loss_his = []
    batch_size = 10
    for iter_num in range(iterations) :
        separation_model.train()
        raw = sample_batch(mus_train, batch_size, 5)
        x, target = transform(raw, hop_length=1024, n_fft=2048)
        x = np.log1p(x)
        x = torch.FloatTensor(x).to(device)
        target = np.log1p(target)
        target = torch.FloatTensor(target).to(device)
        
        y_hat = separation_model(x)
        loss = criterion(y_hat, target)
        gru_optim.zero_grad()
        conv_optim.zero_grad()
        loss.backward()
        gru_optim.step()
        conv_optim.step()
        loss_his.append(loss.item())
    
        if (iter_num + 1) % 10 == 0 :
            separation_model.eval()
            with torch.no_grad() :
                raw_val = sample_batch(mus_val, 10, 5)
                x_val, target_val = transform(raw_val, 1024, 2048)
                x_val = np.log1p(x)
                x_val = torch.FloatTensor(x).to(device)
                target_val = np.log1p(target_val)
                target_val = torch.FloatTensor(target).to(device)
                loss_val = criterion(x_val, target_val)
                print("Iter : {} Loss : {} Loss_val : {}".format(iter_num + 1, loss.item(), loss_val.item()))



