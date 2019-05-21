import librosa
import numpy as np

def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    '''
    Method to convert a list of songs to a np array of melspectrograms
    '''
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
        hop_length = hop_length)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))

def splitsongs(X, y, window = 0.1, overlap = 0.5):
    '''
    Method to split a song into multiple songs using overlapping windows
    '''
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))

    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)

def normalize(img, mean, std):
    img = img/255.0
    img[0] = (img[0] - mean[0]) / std[0]
    img[1] = (img[1] - mean[1]) / std[1]
    img[2] = (img[2] - mean[2]) / std[2]
    img = np.clip(img, 0.0, 1.0)

    return img

# img = normalize(img, mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
