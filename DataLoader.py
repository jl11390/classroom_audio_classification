import pickle
import os
import numpy as np
from public_func import get_local_rms_max
from FeatureExtract import AudioFeature
from AudioSplitter import AudioSplitter


# data loader for training
class DataLoader:
    def __init__(self, file_name, audio_path, cache_path, metadict, frac_t, long_frac_t, long_long_frac_t, step_t,
                 target_class_version=0):
        self.file_name = file_name  # wav file name
        self.audio_path = audio_path  # path that contains the wav files
        self.cache_path = cache_path  # path that contains the preprocessed features
        self.metadict = metadict  # annotation dict for the wav file

        self.frac_t = frac_t
        self.long_frac_t = long_frac_t
        self.long_long_frac_t = long_long_frac_t
        self.step_t = step_t

        self.feature_path = os.path.join(self.cache_path, self.file_name.replace('.wav', '.pkl'))
        self.file_path = os.path.join(self.audio_path, self.file_name)

        self.target_class_version = target_class_version

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
            audiosplitter = AudioSplitter(self.file_path, self.metadict, target_class_version=self.target_class_version)
            audiosplitter.split_audio(self.frac_t, self.long_frac_t, self.long_long_frac_t, self.step_t, threshold=0.3)
            audiosplitter.remove_noisy_data(remove_no_label_data=True, remove_transition=False)
            datas, long_datas, long_long_datas, labels_matrix = audiosplitter.datas, audiosplitter.long_datas, audiosplitter.long_long_datas, audiosplitter.labels
            for bundle in zip(datas, long_datas, long_long_datas):
                data, long_data, long_long_data = bundle
                audio_feature = AudioFeature(data)
                audio_long_feature = AudioFeature(long_data)
                audio_long_long_feature = AudioFeature(long_long_data)
                audio_feature.extract_features(['mfcc', 'spectral', 'rms'])
                audio_long_feature.extract_features(['mfcc', 'spectral', 'rms'])
                audio_long_long_feature.extract_features(['mfcc', 'spectral', 'rms'])
                assert audio_feature.features.shape == audio_long_feature.features.shape and audio_long_feature.features.shape == audio_long_long_feature.features.shape
                audio_diff_features_1 = audio_long_feature.features - audio_feature.features
                audio_diff_features_2 = audio_long_long_feature.features - audio_feature.features
                audio_final_features = np.concatenate(
                    (audio_feature.features, audio_diff_features_1, audio_diff_features_2))
                features_matrix = np.vstack(
                    [features_matrix, audio_final_features]) if features_matrix is not None else audio_final_features
            local_rms_max = get_local_rms_max(audiosplitter.y)
            local_rms_max_feature = np.full((labels_matrix.shape[0], 1), local_rms_max)
            features_matrix = np.hstack([features_matrix, local_rms_max_feature])
            self.save_pickle([features_matrix, labels_matrix], self.feature_path)
            print(f'features and labels extracted and cached successfully from {self.file_name}')
        return features_matrix, labels_matrix


if __name__ == "__main__":
    file_name, audio_path, cache_path, frac_t, long_frac_t, long_long_frac_t, step_t = '48ad890d-ActiveLearning_6.wav', 'data/COAS_2/Audios', 'data/COAS_2/Features', 5, 20, 60, 2
    metadict = {
        "video_url": "/data/upload/3/48ad890d-ActiveLearning_6.mp4",
        "id": 137,
        "tricks": [
            {
                "start": 0,
                "end": 135.59653630013878,
                "labels": [
                    "Other"
                ]
            },
            {
                "start": 135.08096011648806,
                "end": 149.2593051668828,
                "labels": [
                    "Lecturing"
                ]
            },
            {
                "start": 149.00151707505745,
                "end": 158.7974645644211,
                "labels": [
                    "Individual Student Work"
                ]
            },
            {
                "start": 158.53967647259574,
                "end": 224.27563988806222,
                "labels": [
                    "Lecturing"
                ]
            },
            {
                "start": 223.50227561258617,
                "end": 275.57547016130866,
                "labels": [
                    "Q/A"
                ]
            },
            {
                "start": 275.57547016130866,
                "end": 307.79898163947854,
                "labels": [
                    "Other"
                ]
            }
        ],
        "annotator": 1,
        "annotation_id": 133,
        "created_at": "2022-10-16T21:54:07.079646Z",
        "updated_at": "2022-10-16T21:54:07.079683Z",
        "lead_time": 174.194
    }

    dataloader = DataLoader(file_name, audio_path, cache_path, metadict, frac_t, long_frac_t, long_long_frac_t, step_t)
    features_matrix, labels_matrix = dataloader.load_data(load_cache=False)
    print(features_matrix.shape, labels_matrix.shape)
    print(np.sum(labels_matrix, axis=0) / np.sum(labels_matrix))
