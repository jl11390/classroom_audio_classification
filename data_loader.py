from feature_extract import AudioFeature
from AudioSplitter import AudioSplitter
import pickle
import os
import numpy as np


class DataLoader:
    def __init__(self, file_name, audio_path, cache_path, metadict, frac_t, step_t):
        self.file_name = file_name
        self.audio_path = audio_path
        self.cache_path = cache_path
        self.metadict = metadict
        self.frac_t = frac_t
        self.step_t = step_t
        self.feature_path = os.path.join(self.cache_path, self.file_name.replace('.wav', '.pkl'))
        self.file_path = os.path.join(self.audio_path, self.file_name)

    def save_pickle(self, features_labels, save_path):
        output_path = 'data/COAS/Features'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(save_path, "wb") as f:
            pickle.dump(features_labels, f)

    def load_data(self, load_cache = False):
        if load_cache and os.path.exists(self.feature_path):
            with open(self.feature_path, 'rb') as f:
                features_labels = pickle.load(f)
            features_matrix, labels_matrix = features_labels
        else:
            features_matrix = None
            labels_matrix = None
            audiosplitter = AudioSplitter(self.file_path, self.metadict, target_class_version=1)
            audiosplitter.split_audio(self.frac_t, self.step_t, threshold=0.3)
            audiosplitter.remove_noisy_data(remove_no_label_data=True, remove_transition=True)
            datas, labels = audiosplitter.datas, audiosplitter.labels
            for bundle in zip(datas, labels):
                data, label = bundle
                audio_feature = AudioFeature(data, label)
                audio_feature.extract_features(['mfcc', 'spectral', 'rms'])
                features_matrix = np.vstack([features_matrix, audio_feature.features]) if features_matrix is not None else audio_feature.features
                labels_matrix = np.vstack([labels_matrix, audio_feature.label]) if labels_matrix is not None else audio_feature.label
            self.save_pickle([features_matrix, labels_matrix], self.feature_path)
        
        return features_matrix, labels_matrix

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
    features_matrix, labels_matrix = dataloader.load_data(load_cache = False)
    print(features_matrix.shape, labels_matrix.shape)