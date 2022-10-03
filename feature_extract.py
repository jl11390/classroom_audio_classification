import librosa
import numpy as np
import json
import pickle
from pathlib import Path
import os


class AudioFeature:
    def __init__(self, y, sr, label, fold):
        self.y, self.sr = y, sr
        self.label, self.fold = label, fold
        self.features = None

    def _concat_features(self, feature):
        """
        Whenever a self._extract_xxx() method is called in this class,
        this function concatenates to the self.features[i] feature vector
        """
        self.features = np.hstack(
            [self.features, feature] if self.features is not None else feature
        )

    def _extract_rms(self):
        # window and hop here is for calculating rms in a splitted audio
        X = librosa.stft(self.y)
        Y = librosa.amplitude_to_db(np.abs(X), ref=np.max)

        rms = librosa.feature.rms(S=Y)
        rms_feature = np.array([rms.mean(), rms.std()])
        # print(rms_feature.shape)
        self._concat_features(rms_feature)

    def _extract_mfcc(self, n_mfcc=20):
        mfcc = librosa.feature.mfcc(self.y, sr=self.sr, n_mfcc=n_mfcc)

        mfcc_mean = mfcc.mean(axis=1).T
        mfcc_std = mfcc.std(axis=1).T
        # mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
        # print(mfcc_feature.shape)
        self._concat_features(mfcc_mean)
        self._concat_features(mfcc_std)


    def _extract_spectral_contrast(self, n_bands=6, fmin=60):
        spec_con = librosa.feature.spectral_contrast(
            y=self.y, sr=self.sr, n_bands=n_bands, fmin=fmin
        )

        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        spec_con_feature = np.hstack([spec_con_mean, spec_con_std])
        # print(spec_con_feature.shape)
        self._concat_features(spec_con_feature)

    def extract_features(self, feature_list):
        """
        Specify a list of features to extract, and a feature vector will be
        built for you for a given Audio sample.
        """
        extract_fn = dict(
            rms=self._extract_rms,
            mfcc=self._extract_mfcc,
            spectral=self._extract_spectral_contrast,
        )

        for feature in feature_list:
            extract_fn[feature]()


if __name__ == "__main__":
    pass