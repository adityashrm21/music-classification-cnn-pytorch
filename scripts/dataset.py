from tqdm import tqdm
import os
import librosa
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from utils import splitsongs, normalize

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

    X_train_rgb = []
    X_test_rgb = []

    for i in range(X_train.shape[0]):
    #     print(X_train[i, :,:,:].shape, X_train[i, :,:,0].shape)
        X_train_sub = np.concatenate((X_train[i, :,:,:], X_train[i, :,:, :].reshape(128, 129, 1)), axis=2)
        X_train_sub = np.concatenate((X_train_sub, X_train[i, :,:, 0].reshape(128, 129, 1)), axis=2)
        X_train_sub = cv2.resize(X_train_sub, (224, 224))
        X_train_sub = normalize(X_train_sub, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        X_train_rgb.append(X_train_sub)

    for i in range(X_test.shape[0]):
        X_test_sub = np.concatenate((X_test[i, :,:,:], X_test[i, :,:, :].reshape(128, 129, 1)), axis=2)
        X_test_sub = np.concatenate((X_test_sub, X_test[i, :,:, 0].reshape(128, 129, 1)), axis=2)
        X_test_sub = cv2.resize(X_test_sub, (224, 224))
        X_test_sub = normalize(X_test_sub, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        X_test_rgb.append(X_test_sub)

    X_train_rgb = np.array(X_train_rgb, dtype=np.float32)
    X_test_rgb = np.array(X_test_rgb, dtype=np.float32)
    # print("shape before reshape", X_train_rgb.shape, X_test_rgb.shape)
    X_train_rgb = X_train_rgb.reshape(-1, 3, 224, 224)

    X_test_rgb = X_test_rgb.reshape(-1, 3, 224, 224)

    return X_train_rgb, X_test_rgb, y_train, y_test
