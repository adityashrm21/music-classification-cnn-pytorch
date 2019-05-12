from tqdm import tqdm
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from utils import splitsongs

def read_data(src_dir, genres, song_samples, spec_format, debug = True):
    # Empty array of dicts with the processed features from all files
    arr_specs = []
    arr_genres = []

    # Read files from the folders
    for x,_ in genres.items():
        folder = src_dir + x

        for root, subdirs, files in tqdm(os.walk(folder)):
            for file in files:
                # Read the audio file
                file_name = folder + "/" + file
                signal, sr = librosa.load(file_name)
                signal = signal[:song_samples]

                # Debug process
                if debug:
                    print("Reading file: {}".format(file_name))

                # Convert to dataset of spectograms/melspectograms
                signals, y = splitsongs(signal, genres[x])

                # Convert to "spec" representation
                specs = spec_format(signals)

                # Save files
                arr_genres.extend(y)
                arr_specs.extend(specs)

    return np.array(arr_specs), np.array(arr_genres)


def get_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5240,
                                                random_state=42, stratify = y)
    X_train = X_train.reshape(-1, 1, 128, 129)

    X_test = X_test.reshape(-1, 1, 128, 129)

    return X_train, X_test, y_train, y_test
