import os
import librosa
import pickle
import numpy as np
from public_func import get_local_rms_max
from FeatureExtract import AudioFeature


class WavToFeatures:
    def __init__(self, file_name, audio_path, cache_path, frac_t, long_frac_t, long_long_frac_t, step_t):
        self.file_name = file_name  # wav file name
        self.audio_path = audio_path  # test audio path that contains the wav files
        self.cache_path = cache_path
        self.frac_t = frac_t
        self.long_frac_t = long_frac_t
        self.long_long_frac_t = long_long_frac_t
        self.step_t = step_t

        self.file_path = os.path.join(self.audio_path, self.file_name)
        self.feature_path = os.path.join(self.cache_path, self.file_name.replace('.wav', '.pkl'))

    def save_pickle(self, features, save_path):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(save_path, "wb") as f:
            pickle.dump(features, f)

    def transform(self, load_cache=False, multi_scaling=True):
        """
        transform a wav file into feature matrix, used for inference
        """
        if load_cache and os.path.exists(self.feature_path):
            with open(self.feature_path, 'rb') as f:
                features = pickle.load(f)
            features_matrix = features
            # print(f'features and labels load successfully for evaluation file {self.file_name}')
        else:
            y, sr = librosa.load(self.file_path, sr=22050, mono=True)
            num_samples = len(y)  # total number of samples
            audio_length = num_samples / sr  # audio length in seconds
            num_samples_frame = self.frac_t * sr  # number of samples in each frame

            n_frames = ((audio_length - self.frac_t) // self.step_t) + 1  # total number of frames that could be extracted
            n_frames = int(n_frames)

            # To get long "multi-scaling"
            long_audio_length_1 = (n_frames - 1) * self.step_t + self.long_frac_t
            long_num_samples_1 = int(np.ceil(long_audio_length_1 * sr))
            num_paddings_1 = long_num_samples_1 - num_samples
            long_num_samples_frame_1 = self.long_frac_t * sr
            long_y = np.pad(y, (0, num_paddings_1), 'mean')

            # To get long-long "multi-scaling"
            long_audio_length_2 = (n_frames - 1) * self.step_t + self.long_long_frac_t
            long_num_samples_2 = int(np.ceil(long_audio_length_2 * sr))
            num_paddings_2 = long_num_samples_2 - num_samples
            long_num_samples_frame_2 = self.long_long_frac_t * sr
            long_long_y = np.pad(y, (0, num_paddings_2), 'mean')

            datas = np.zeros((n_frames, num_samples_frame))  # contains the frames extracted from audio
            long_datas = np.zeros((n_frames, long_num_samples_frame_1))
            long_long_datas = np.zeros((n_frames, long_num_samples_frame_2))

            for i in range(n_frames):
                left_t = i * self.step_t
                right_t = i * self.step_t + self.frac_t
                long_right_t = i * self.step_t + self.long_frac_t
                long_long_right_t = i * self.step_t + self.long_long_frac_t
                datas[i] = y[left_t * sr:right_t * sr]
                long_datas[i] = long_y[left_t * sr:long_right_t * sr]
                long_long_datas[i] = long_long_y[left_t * sr:long_long_right_t * sr]

            features_matrix = None

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
                audio_final_features = np.concatenate((audio_feature.features, audio_diff_features_1, audio_diff_features_2))
                features_matrix = np.vstack(
                    [features_matrix, audio_final_features]) if features_matrix is not None else audio_final_features
            local_rms_max = get_local_rms_max(y)
            local_rms_max_feature = np.full((n_frames, 1), local_rms_max)
            features_matrix = np.hstack([features_matrix, local_rms_max_feature])
            self.save_pickle(features_matrix, self.feature_path)
            print(f'successfully transformed the wav file {self.file_name} into a {features_matrix.shape} matrix')
        if not multi_scaling:
            original_data_dim = int((features_matrix.shape[1] - 1) / 3)
            fancy_index = [k for k in range(original_data_dim)]
            fancy_index.append(-1)
            features_matrix = features_matrix[:, fancy_index]
        return features_matrix


if __name__ == "__main__":
    file_name, audio_path, cache_path, frac_t, long_frac_t, long_long_frac_t, step_t = '48ad890d-ActiveLearning_6.wav', 'data/COAS_2/Audios', 'data/COAS_2/Features_test', 5, 20, 60, 2
    wav_to_features = WavToFeatures(file_name, audio_path, cache_path, frac_t, long_frac_t, long_long_frac_t, step_t).transform(load_cache=False, multi_scaling=True)
    print(wav_to_features.shape)
