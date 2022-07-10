from transformers import logging

import glob
import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

logging.set_verbosity_error()

audios = glob.glob('../fma_medium/*/*.mp3')
print(len(audios))

output_root = 'spectrograms'


def save_mel(path, output_path):
    if os.path.exists(output_path): 
        return
    try:
        audio_data, sr = librosa.load(path, sr=16000)
        plt.figure(figsize=(3,3))
        M = librosa.feature.melspectrogram(y=audio_data, sr=sr, win_length = int(len(audio_data)/512), hop_length = int(len(audio_data)/2048))
        M_db = librosa.power_to_db(M, ref=np.max)
        librosa.display.specshow(M_db, sr = sr, hop_length = int(len(audio_data)/2048), y_axis='mel', x_axis='time')
        plt.axis("off")
        plt.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.close() 
    except:
        with open('error.txt', 'a+') as f:
            f.write(path)


Parallel(n_jobs=12)(delayed(save_mel)(audio, output_root + '/' + os.path.split(audio)[1][:-4] + '.jpeg') for audio in tqdm(audios))



