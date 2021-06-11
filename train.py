from torch.nn.modules import conv
from torch.optim import Adam
import torch
import numpy as np
from utils import conv_params, gru_params, sample_batch, transform
from model import ARC, Enhancement
import torch.nn as nn
import musdb

def train_ARC(iter_num, batch_size, dset_train, dset_val, gru_lr = 1e-4, conv_lr = 1e-3, device='cpu') :
    """
        Traning separation model
    """
    train_his = []
    device = torch.device(device)
    model = ARC()
    gru_optimizer = Adam(gru_params(model), lr = gru_lr)
    conv_optimizer = Adam(conv_params(model), lr = conv_lr)
    criterion = nn.MSELoss()
    for iter in range(0, iter_num) :
        model.train()
        raw = sample_batch(dset_train, batch_size, 5)
        x, target = transform(raw, hop_length=1024, n_fft=2048)
        x = np.log1p(x)
        x = torch.FloatTensor(x).to(device)
        target = np.log1p(target)
        y_hat = model(x)
        gru_optimizer.zero_grad()
        conv_optimizer.zero_grad()
        loss = criterion(y_hat, target)
        loss.backward()
        gru_optimizer.step()
        conv_optimizer.step()
        train_his.append(loss.item())
        if (iter + 1) % 10 == 0 :
            model.eval()
            with torch.no_grad() :
                raw_val = sample_batch(dset_val, batch_size, 5)
                x_val, target_val = transform(raw_val, 1024, 2048)
                x_val = np.log1p(x_val)
                x_val = torch.FloatTensor(x_val).to(device)
                target_val = np.log1p(target_val)
                target_val = torch.FloatTensor(target_val).to(device)
                y_hat_val = model(x_val)
                loss_val = criterion(y_hat_val, target_val)
            print("Iterations : {} Loss : {:.4f}  Val_Loss : {:.4f} ".format(iter+1, loss.item(), loss_val.item()))
    torch.save(model.state_dict(), 'ARC.pt')
    return model, train_his

def train_Enhancement(ARC_model, iter_num, batch_size, dset_train, dset_val, device, mode) :
    """
        Training enhancement model
        ARC_model : separation model 
        mode : drum or bass or other or vocal
    """
    if mode == 'drum' :
        id = 0
    elif mode == 'bass' :
        id = 1
    elif mode == 'other' :
        id = 2
    elif mode == 'vocal' :
        id = 3
    else :
        raise "ModeError"
    model = Enhancement()
    optimizer = Adam(model.parameters())
    criterion = nn.MSELoss()
    ARC_model.eval()
    train_his = []
    for iter in range(iter_num) :
        model.train()
        raw = sample_batch(dset_train, batch_size, 5)
        x, target = transform(raw, hop_length=1024, n_fft=2048)
        x = np.log1p(x)
        x = torch.FloatTensor(x).to(device)
        target = np.log1p(target)[:, 1025*id : 1025*(id+1), :]
        with torch.no_grad() :
            separation = ARC_model(x)[:, 1025*id : 1025*(id+1), :]
        y_hat = model(separation)
        optimizer.zero_grad()
        loss = criterion(y_hat, target)
        loss.backward()
        optimizer.step()
        train_his.append(loss.item())
        if (iter + 1) % 10 == 0 :
            model.eval()
            with torch.no_grad() :
                raw_val = sample_batch(dset_val, batch_size, 5)
                x_val, target_val = transform(raw_val, 1024, 2048)
                x_val = np.log1p(x_val)
                x_val = torch.FloatTensor(x_val).to(device)
                target_val = np.log1p(target_val)[:, 1025*id : 1025*(id+1), :]
                target_val = torch.FloatTensor(target_val).to(device)
                separation_val = ARC_model(x_val)[:, 1025*id : 1025*(id+1), :]
                y_hat_val = model(separation_val)
                loss_val = criterion(y_hat_val, target_val)
            print("Iterations : {} {}_Loss : {:.4f}  {}_Val_Loss : {:.4f} ".format(iter+1, mode, loss.item(), mode, loss_val.item()))
    path = mode + '.pt'
    torch.save(model.state_dict(), path)
    return model, train_his

if __name__ == '__main__' :
    mus_train = musdb.DB('./musdb18', subsets = 'train')
    mus_val = musdb.DB('./musdb18', subsets = 'test')
    ARC_model, ARC_his = train_ARC(10000, 10, mus_train, mus_val)
    Bass_enhancement, Bass_his = train_Enhancement(ARC_model, 10000, 10, mus_train, mus_val, device = 'cpu', mode = 'bass')
    Vocal_enhancement, Vocal_his = train_Enhancement(ARC_model, 10000, 10, mus_train, mus_val, device = 'cpu', mode = 'vocal')
    Drums_enhancement, Drum_his = train_Enhancement(ARC_model, 10000, 10, mus_train, mus_val, device = 'cpu', mode = 'drum')
    Other_enhancement, Other_his = train_Enhancement(ARC_model, 10000, 10, mus_train, mus_val, device = 'cpu', mode = 'other')