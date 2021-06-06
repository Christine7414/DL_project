import musdb
import numpy as np
import random
from scipy import signal
import matplotlib.pyplot as plt
import torch
def sample_batch(dset, batch_size) :
    idx = np.random.randint(0, 100, (4, batch_size))
    # bs x sample x 2
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

        drum_track.chunk_duration = 5
        bass_track.chunk_duration = 5
        other_track.chunk_duration = 5
        vocal_track.chunk_duration = 5

        drum_track.chunk_start = random.uniform(0, drum_track.duration - drum_track.chunk_duration)
        bass_track.chunk_start = random.uniform(0, bass_track.duration - bass_track.chunk_duration)
        other_track.chunk_start = random.uniform(0, other_track.duration - other_track.chunk_duration)
        vocal_track.chunk_start = random.uniform(0, vocal_track.duration - vocal_track.chunk_duration)

        drum[i] = drum_track.targets['drums'].audio.T
        bass[i] = bass_track.targets['bass'].audio.T
        other[i] = other_track.targets['other'].audio.T
        vocal[i] = vocal_track.targets['vocals'].audio.T
        mix[i] = drum[i] + bass[i] + other[i] + mix[i]
    return {'drum' : drum,
            'bass' : bass,
            'other' : other,
            'vocal' : vocal,
            'mix' : mix}


def gru_params(model) :
    for name, params in model.named_parameters() :
        if(name[:3] == 'gru') :
            yield params
def conv_params(model) :
    for name, params in model.named_parameters() :
        if(name[:3] != 'gru') :
            yield params