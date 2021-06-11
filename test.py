import musdb
from train import ARC_model
import librosa
import numpy as np
import torch
import norbert

def stft_spec(track, start) :
    """
        Input : 
            taack : track in musdb18
            start : start of track (in second)
        Return :
            stft1 : stft of channel1 of track
            stft2 : stft of channel2 of track
            x1 : spectrogram of channel1 of track
            x2 : spectrogram of channel2 of track
    """
    x = track.audio.T[0, 44100*start : 44100*(start+5)]
    stft1 = librosa.stft(x, n_fft=2048, hop_length=1024)
    x = track.audio.T[1, 44100*start : 44100*(start+5)]
    stft2 = librosa.stft(x, n_fft=2048, hop_length=1024)
    x1 = np.abs(stft1)
    x1 = np.log1p(x1)
    x1 = torch.FloatTensor(x1).unsqueeze(0)
    x2 = np.abs(stft2)
    x2 = np.abs(x2)
    x2 = torch.FloatTensor(x2).unsqueeze(0)
    return stft1, stft2, x1, x2

def source_split(spec):
    """
        Input : Spectrogram of sources with shape (num_samples, num_frames*4, num_bins)
        Return : Spectrogram of drum, bass, other, vocal
    """
    d = spec[:, :1025, :]
    b = spec[:, 1025 : 1025*2, :]
    o = spec[:, 1025*2: 1025*3, :]
    v = spec[:, 1025*3 :, :]
    return d, b, o, v

def get_frame(y1, y2) :
    """
        Input : 
            y1 : spectrogram of sources of channel1 with shape (num_samples, num_frames*4, num_bins)
            y2 : spectrogram of sources of channel2 with shape (num_samples, num_frames*4, num_bins)
        Return :
            y : spectrogram of sources with shape (nb_frames, nb_bins, 2, nb_sources)
    """
    d1, b1, o1, v1 = source_split(y1)
    d2, b2, o2, v2 = source_split(y2)
    d = torch.cat((d1,d2), dim = 0).permute(1,2,0)
    b = torch.cat((b1,b2), dim = 0).permute(1,2,0)
    o = torch.cat((o1,o2), dim = 0).permute(1,2,0)
    v = torch.cat((v1,v2), dim = 0).permute(1,2,0)
    y = torch.stack((d, b, o, v), dim = -1)
    return y

def get_part(stft, part) :
    """
        Input : 
            stft : Sort-time Fouier transform of part with shape (num_frames, num_bins, num_channels, nb_sources)
            part : drum or bass or other or vocal
        Return :
            Raw audio signal for part
    """
    b = stft[:, :, :, 1] 
    b = np.stack((librosa.istft(b[:, :, 0], hop_length=1024, length=44100*5), librosa.istft(b[:, :, 1], hop_length=1024, length=44100*5)), axis = -1)
    d = stft[:, :, :, 0]
    d = np.stack((librosa.istft(d[:, :, 0], hop_length=1024, length=44100*5), librosa.istft(d[:, :, 1], hop_length=1024, length=44100*5)), axis = -1)
    o = stft[:, :, :, 2]
    o = np.stack((librosa.istft(o[:, :, 0], hop_length=1024, length=44100*5), librosa.istft(o[:, :, 1], hop_length=1024, length=44100*5)), axis = -1)
    v = stft[:, :, :,3]
    v = np.stack((librosa.istft(v[:, :, 0], hop_length=1024, length=44100*5), librosa.istft(v[:, :, 1], hop_length=1024, length=44100*5)), axis = -1)
    if part == 'drum' :
        return d
    if part == 'bass' :
        return b
    if part == 'other' :
        return o
    if part == 'vocal' :
        return v
    if part == 'accompany' :
        return d+b+o

def separation(model, track, start, part) :
    """
        Input :
            model : ARC model
            start : starting point in second
        Return :
            Audio signal of track from stating point start of part
    """
    model.eval()
    stft1, stft2, x1, x2 = stft_spec(track, start)
    with torch.no_grad() :
        y1 = model(x1)
        y2 = model(x2)
    V = get_frame(y1, y2).cpu().numpy()
    stft = np.stack((stft1, stft2), axis = -1)
    Y = norbert.wiener(V, stft)
    audio = get_part(Y, part)
    return audio

if __name__ == '__main__' :
    mus = musdb.DB('./musdb18')
    part = 'bass'
    track = mus[15]
    start = 30
    model = ARC_model()
    model.load_state_dict(torch.load('./ARC.pt'))
    raw_audio = separation(model, track, start, part)


