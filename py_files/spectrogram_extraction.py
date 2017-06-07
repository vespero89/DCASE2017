#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: daniele
"""

#import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import librosa
from os import walk, path


def wav_file_list(source):
    # list all file in source directory
    filenames = []
    for (dirpath, dirnames,  filenames) in walk(source):
        break
    # drop all non wav file
    wav_filenames = [f for f in filenames if f.lower().endswith('.wav')]
    
    return wav_filenames


# calcola uno spettrogramma
def spectrogram(filepath, fs, N, overlap, win_type='hamming'):
    
    # Load an audio file as a floating point time series
    x, fs  = librosa.core.load(filepath,sr=fs)
    # Returns: np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype], dtype=64-bit complex
    X = librosa.core.stft(x, n_fft=N, window=signal.get_window(win_type,N), hop_length=N-overlap, center=False)
    #Sxx = np.abs(X)**2
    Sxx = librosa.logamplitude(np.abs(X)**2,ref_power=np.max)

    return Sxx


# estrae gli spettrogrammi dai file contenuti in source e li salva in dest
def extract_spectrograms(source, dest, fs, N, overlap, win_type='hamming'):

    wav_filenames = wav_file_list(source)
    for w in wav_filenames:
        Sxx=spectrogram(path.join(source,w), fs, N, overlap, win_type)
        np.save(path.join(dest,w[0:-4]),Sxx)


# calcola i mel, i delta e i delta-deltas
def log_mel(filepath, fs, N, overlap, win_type='hamming', n_mels=128, fmin=0.0, fmax=None, htk=True, delta_width=2):

    coefficients = []
    # Load an audio file as a floating point time series
    x, fs  = librosa.core.load(filepath,sr=fs)
    # Power spectrum
    S = np.abs(librosa.core.stft(x, n_fft=N, window=signal.get_window(win_type,N), hop_length=N-overlap, center=False))**2
    # Build a Mel filter
    mel_basis = librosa.filters.mel(fs, N, n_mels, fmin, fmax, htk)
    # Filtering
    mel_filtered = np.dot(mel_basis, S)

    coefficients.append(mel_filtered)
    # add delta e delta-deltas
    coefficients.append(librosa.feature.delta(mel_filtered, delta_width*2+1, order=1, axis=-1))
    coefficients.append(librosa.feature.delta(mel_filtered, delta_width*2+1, order=2, axis=-1))
        
    return coefficients


# estrae i mel e ci calcola i delta e i delta-deltas dai file contenuti in source e li salva in dest
# per la versione log è sufficiente eseguire librosa.logamplitude(.) alla singola sottomatrice o all'intera matrice
def extract_log_mel(source, dest, fs, N, overlap, win_type='hamming', n_mels=128, fmin=0.0, fmax=None, htk=True, delta_width=2):
    
    wav_filenames = wav_file_list(source)
    for w in wav_filenames:
        mels=log_mel(path.join(source,w), fs, N, overlap, win_type, n_mels, fmin, fmax, htk, delta_width)
        np.save(path.join(dest,w[0:-4]),mels)


if __name__ == "__main__":

    root_dir = path.realpath('..')

    wav_dir_path = path.join(root_dir,'dataset','all_file')
    dest_path=path.join(root_dir,'dataset','spectrograms')
    dest_path_log_mel=path.join(root_dir,'dataset','mel_coefficients')
    window_type = 'hamming'
    fft_length = 256
    overlap = 128
    Fs = 8000    
    n_mels=29
    fmin=0.0
    fmax=Fs/2
    htk=True
    delta_width=2
    
    extract_spectrograms(wav_dir_path, dest_path, Fs, fft_length, overlap, 'hamming')
    extract_log_mel(wav_dir_path, dest_path_log_mel, Fs, fft_length, overlap, window_type, n_mels, fmin, fmax, htk, delta_width)
    
    import os
    print(os.path.realpath('.'))
