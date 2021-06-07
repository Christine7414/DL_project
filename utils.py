import musdb
import numpy as np
import random
from scipy import signal
import matplotlib.pyplot as plt
import torch
from librosa import stft
def sample_batch(dset, batch_size, second) :
    """
        dset : musdb dataset
        second : length of sub-clip of songs in second
    """
    idx = np.random.randint(0, 100, (4, batch_size))
    drum = np.zeros((batch_size, 2, 5 * 44100))
    bass = np.zeros((batch_size, 2, 5 * 44100))
    other = np.zeros((batch_size, 2, 5 * 44100))
    vocal = np.zeros((batch_size, 2, 5 * 44100))
    mix = np.zeros((batch_size, 2, 5 * 44100))
    batch_idx = np.arange(batch_size)
    for drum_idx, bass_idx, other_idx, vocal_idx, i in zip(idx[0], idx[1], idx[2], idx[3], batch_idx) :

        drum_track = dset[drum_idx]
        bass_track = dset[bass_idx]
        other_track = dset[other_idx]
        vocal_track = dset[vocal_idx]

        drum_track.chunk_duration = second
        bass_track.chunk_duration = second
        other_track.chunk_duration = second
        vocal_track.chunk_duration = second

        drum_track.chunk_start = random.uniform(0, drum_track.duration - drum_track.chunk_duration)
        bass_track.chunk_start = random.uniform(0, bass_track.duration - bass_track.chunk_duration)
        other_track.chunk_start = random.uniform(0, other_track.duration - other_track.chunk_duration)
        vocal_track.chunk_start = random.uniform(0, vocal_track.duration - vocal_track.chunk_duration)

        drum[i] = drum_track.targets['drums'].audio.T
        bass[i] = bass_track.targets['bass'].audio.T
        other[i] = other_track.targets['other'].audio.T
        vocal[i] = vocal_track.targets['vocals'].audio.T
        mix[i] = drum[i] + bass[i] + other[i] + vocal[i]
    return {'drum' : drum,
            'bass' : bass,
            'other' : other,
            'vocal' : vocal,
            'mix' : mix}

def transform(raw, hop_length, n_fft, mono = False) :
    """
    generaing spectrogram from raw audio signal
    input :
        raw : time domain audio signal (bs x channel x samples)
        n_fft = length of stft
        hop_length : hop length of stft
    output :
        x : spectrogram of mixture signal (bs*2 x freq_bin x time_bim)
        target : concatenation of spectrogram of drums, bass, other, vocals (bs*2 x freq_bin*4 x time_bim)
    """
    channel = 1 if mono else 2
    x = stft(raw['mix'], n_fft = n_fft, hop_length = hop_length)
    x = np.asarray([stft(raw['mix'][i][j], n_fft=2048, hop_length=1024) for j in range(channel) for i in range(10)])
    drum_stft = np.asarray([stft(raw['drum'][i][j], n_fft = n_fft, hop_length = hop_length) for j in range(channel) for i in range(10)])
    bass_stft = np.asarray([stft(raw['bass'][i][j], n_fft = n_fft, hop_length = hop_length) for j in range(channel) for i in range(10)])
    other_stft = np.asarray([stft(raw['other'][i][j], n_fft = n_fft, hop_length = hop_length) for j in range(2) for i in range(10)])
    vocal_stft = np.asarray([stft(raw['vocal'][i][j], n_fft = n_fft, hop_length = hop_length) for j in range(2) for i in range(10)])
    target = np.concatenate((drum_stft, bass_stft, other_stft, vocal_stft), axis = 1)
    x = np.abs(x)
    target = np.ans(target)
    return x, target

def gru_params(model) :
    for name, params in model.named_parameters() :
        if(name[:3] == 'gru') :
            yield params
def conv_params(model) :
    for name, params in model.named_parameters() :
        if(name[:3] != 'gru') :
            yield params