import librosa
import numpy as np
import pickle
from pathlib import Path
import os


class AudioFeature:
    def __init__(self, src_path, label, fold):
        self.src_path = src_path
        self.label = label
        self.fold = fold
        self.y, self.sr = librosa.load(src_path, mono=True, sr=None)
        self.features = None

    def _concat_features(self, feature):
        """
        Whenever a self._extract_xxx() method is called in this class,
        this function concatenates to the self.features feature vector
        """
        self.features = np.hstack(
            [self.features, feature] if self.features is not None else feature
        )

    def _extract_rms(self, window_t=10, hop_t=1):
        X = librosa.stft(self.y, n_fft=window_t*self.sr, hop_length=hop_t*self.sr, win_length=window_t*self.sr, window='hanning')
        Y = librosa.amplitude_to_db(np.abs(X), ref=np.max)

        rms = librosa.feature.rms(S=Y, frame_length=window_t*self.sr, hop_length=hop_t*self.sr)
        rms_feature = np.array([rms.mean(), rms.std()])
        # print(rms_feature.shape)
        self._concat_features(rms_feature)

    def _extract_mfcc(self, n_mfcc=25):
        mfcc = librosa.feature.mfcc(self.y, sr=self.sr, n_mfcc=n_mfcc)

        mfcc_mean = mfcc.mean(axis=1).T
        mfcc_std = mfcc.std(axis=1).T
        mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
        # print(mfcc_feature.shape)
        self._concat_features(mfcc_feature)

    def _extract_spectral_contrast(self, n_bands=3):
        spec_con = librosa.feature.spectral_contrast(
            y=self.y, sr=self.sr, n_bands=n_bands
        )

        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        spec_con_feature = np.hstack([spec_con_mean, spec_con_std])
        # print(spec_con_feature.shape)
        self._concat_features(spec_con_feature)

    def _extract_chroma_stft(self):
        stft = np.abs(librosa.stft(self.y))
        chroma_stft = librosa.feature.chroma_stft(S=stft, sr=self.sr)
        chroma_mean = chroma_stft.mean(axis=1).T
        chroma_std = chroma_stft.std(axis=1).T
        chroma_feature = np.hstack([chroma_mean, chroma_std])
        # print(chroma_feature.shape)
        self._concat_features(chroma_feature)

    def extract_features(self, feature_list, save_local=True):
        """
        Specify a list of features to extract, and a feature vector will be
        built for you for a given Audio sample.
        By default the extracted feature and class attributes will be saved in
        a local directory. This can be turned off with save_local=False.
        """
        extract_fn = dict(
            rms=self._extract_rms,
            mfcc=self._extract_mfcc,
            spectral=self._extract_spectral_contrast,
            chroma=self._extract_chroma_stft,
        )

        for feature in feature_list:
            print(f'extracting feature {feature}')
            extract_fn[feature]()

        if save_local:
            self._save_local()

    def _save_local(self, clean_source=True):
        out_name = self.src_path.split("/")[-1]
        out_name = out_name.replace(".wav", "")

        filename = f"data/{out_name}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

        if clean_source:
            self.y = None

if __name__ == "__main__":
    src_path = f"data/test.wav"
    label = 'label1'
    fold = 1
    audio = AudioFeature(src_path, label, fold)
    audio.extract_features(["rms", "mfcc", "spectral"])

    print(audio.features.shape)