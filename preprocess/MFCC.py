import librosa
import numpy as np


class MFCC():
    def __init__(self, n_mfcc, max_samples, n_ftt, hop_length):
        self.n_mfcc = n_mfcc
        self.max_samples = max_samples
        self.n_ftt = n_ftt
        self.hop_length = hop_length

    def _pad_signal(self, y):
        """
        Pads the input signal to match the max_length.
        """
        if len(y) < self.max_samples:
            # Pad the signal with zeros
            padding = self.max_samples - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        return y

    def caclulate(self, input_file):
        y, sr = librosa.load(input_file, sr=None)
        y = self._pad_signal(y)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_ftt, hop_length=self.hop_length)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        return mfccs, mfcc_delta, mfcc_delta2



