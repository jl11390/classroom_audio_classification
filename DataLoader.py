from FeatureExtract import AudioFeature
from AudioSplitter import AudioSplitter
import librosa
import pickle
import os
import numpy as np


# data loader for training
class DataLoader:
    def __init__(self, file_name, audio_path, cache_path, metadict, frac_t, step_t, target_class_version=1):
        self.file_name = file_name  # wav file name
        self.audio_path = audio_path  # path that contains the wav files
        self.cache_path = cache_path  # path that contains the preprocessed features
        self.metadict = metadict  # annotation dict for the wav file

        self.frac_t = frac_t
        self.step_t = step_t

        self.feature_path = os.path.join(self.cache_path, self.file_name.replace('.wav', '.pkl'))
        self.file_path = os.path.join(self.audio_path, self.file_name)

        self.target_class_version = target_class_version

        self.label_dict = None

    def save_pickle(self, features_labels, save_path):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(save_path, "wb") as f:
            pickle.dump(features_labels, f)

    def load_data(self, load_cache=False):
        if load_cache and os.path.exists(self.feature_path):
            with open(self.feature_path, 'rb') as f:
                features_labels = pickle.load(f)
            features_matrix, labels_matrix = features_labels
            print(f'features and labels load successfully for {self.file_name}')
        else:
            features_matrix = None
            labels_matrix = None
            audiosplitter = AudioSplitter(self.file_path, self.metadict, target_class_version=self.target_class_version)
            self.label_dict = audiosplitter.label_dict
            audiosplitter.split_audio(self.frac_t, self.step_t, threshold=0.3)
            audiosplitter.remove_noisy_data(remove_no_label_data=True, remove_transition=False)
            datas, labels = audiosplitter.datas, audiosplitter.labels
            for bundle in zip(datas, labels):
                data, label = bundle
                assert np.sum(label) > 0, 'wrong'
                audio_feature = AudioFeature(data, label)
                audio_feature.extract_features(['mfcc', 'spectral', 'rms'])
                features_matrix = np.vstack(
                    [features_matrix,
                     audio_feature.features]) if features_matrix is not None else audio_feature.features
                labels_matrix = np.vstack(
                    [labels_matrix, audio_feature.label]) if labels_matrix is not None else audio_feature.label
            self.save_pickle([features_matrix, labels_matrix], self.feature_path)
            print(f'features and labels extracted and cached successfully from {self.file_name}')
        return features_matrix, labels_matrix


# data loader for inference
def wav_transform(src_path, frac_t, step_t):
    """
    transform a wav file into feature matrix, used for inference
    """
    y, sr = librosa.load(src_path, sr=22050, mono=True)
    num_samples = len(y)  # total number of samples
    audio_length = num_samples / sr  # audio length in seconds
    num_samples_frame = frac_t * sr  # number of samples in each frame

    n_frames = ((audio_length - frac_t) // step_t) + 1  # total number of frames that could be extracted
    n_frames = int(n_frames)

    datas = np.zeros((n_frames, num_samples_frame))  # contains the frames extracted from audio
    for i in range(n_frames):
        left_t = i * step_t
        right_t = i * step_t + frac_t
        datas[i] = y[left_t * sr:right_t * sr]
    features_matrix = None
    for data in datas:
        audio_feature = AudioFeature(data, None)
        audio_feature.extract_features(['mfcc', 'spectral', 'rms'])
        features_matrix = np.vstack(
            [features_matrix,
             audio_feature.features]) if features_matrix is not None else audio_feature.features
    print(f'successfully transformed the wav file into a {features_matrix.shape} matrix')
    return features_matrix


if __name__ == "__main__":
    file_name, audio_path, cache_path, frac_t, step_t = 'Technology_1_008.wav', 'data/COAS/Audios', 'data/COAS/Features', 10, 2
    metadict = {
        "video_url": "/data/upload/3/b3b9ac82-Technology_1_008.mp4",
        "id": 59,
        "tricks": [
            {
                "start": 71.5442350718065,
                "end": 90.58838397581255,
                "labels": [
                    "Collaborative Student Work"
                ]
            },
            {
                "start": 89.55897052154195,
                "end": 267.64749811035523,
                "labels": [
                    "Lecturing"
                ]
            },
            {
                "start": 267.13279138321997,
                "end": 346.39762736205597,
                "labels": [
                    "Q/A"
                ]
            },
            {
                "start": 344.85350718065007,
                "end": 451.3977996976568,
                "labels": [
                    "Lecturing"
                ]
            }
        ],
        "annotator": 1,
        "annotation_id": 55,
        "created_at": "2022-10-10T16:49:13.958516Z",
        "updated_at": "2022-10-10T16:49:13.958577Z",
        "lead_time": 155.885
    }

    dataloader = DataLoader(file_name, audio_path, cache_path, metadict, frac_t, step_t)
    features_matrix, labels_matrix = dataloader.load_data(load_cache=False)
    print(features_matrix.shape, labels_matrix.shape)
    print(np.sum(labels_matrix, axis=0) / np.sum(labels_matrix))
