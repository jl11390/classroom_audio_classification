import os
import librosa
import numpy as np
from FeatureExtract import AudioFeature


class WavToFeatures:
    def __init__(self, file_name, audio_path, frac_t, long_frac_t, step_t):
        self.file_name = file_name  # wav file name
        self.audio_path = audio_path  # test audio path that contains the wav files

        self.frac_t = frac_t
        self.long_frac_t = long_frac_t
        self.step_t = step_t

        self.file_path = os.path.join(self.audio_path, self.file_name)

    def transform(self):
        """
        transform a wav file into feature matrix, used for inference
        """
        y, sr = librosa.load(self.file_path, sr=22050, mono=True)
        num_samples = len(y)  # total number of samples
        audio_length = num_samples / sr  # audio length in seconds
        num_samples_frame = self.frac_t * sr  # number of samples in each frame

        n_frames = ((audio_length - self.frac_t) // self.step_t) + 1  # total number of frames that could be extracted
        n_frames = int(n_frames)

        # To get "multi-scaling"
        long_audio_length = (n_frames - 1) * self.step_t + self.long_frac_t
        long_num_samples = int(np.ceil(long_audio_length * sr))
        num_paddings = long_num_samples - num_samples
        long_num_samples_frame = self.long_frac_t * sr
        long_y = np.pad(y, (0, num_paddings), 'mean')

        datas = np.zeros((n_frames, num_samples_frame))  # contains the frames extracted from audio
        long_datas = np.zeros((n_frames, long_num_samples_frame))

        for i in range(n_frames):
            left_t = i * self.step_t
            right_t = i * self.step_t + self.frac_t
            long_right_t = i * self.step_t + self.long_frac_t
            datas[i] = y[left_t * sr:right_t * sr]
            long_datas[i] = long_y[left_t * sr:long_right_t * sr]

        features_matrix = None

        for bundle in zip(datas, long_datas):
            data, long_data = bundle
            audio_feature = AudioFeature(data, label=None)
            audio_long_feature = AudioFeature(long_data, label=None)
            audio_feature.extract_features(['mfcc', 'spectral', 'rms'])
            audio_long_feature.extract_features(['mfcc', 'spectral', 'rms'])
            assert audio_feature.features.shape == audio_long_feature.features.shape
            audio_diff_features = audio_long_feature.features - audio_feature.features
            audio_final_features = np.concatenate((audio_feature.features, audio_diff_features))
            features_matrix = np.vstack(
                [features_matrix, audio_final_features]) if features_matrix is not None else audio_final_features
        print(f'successfully transformed the wav file into a {features_matrix.shape} matrix')
        return features_matrix


if __name__ == "__main__":
    file_name, audio_path, frac_t, long_frac_t, step_t = 'Games_4.wav', 'data/COAS/Audios_test', 5, 20, 2
    wav_to_features = WavToFeatures(file_name, audio_path, frac_t, long_frac_t, step_t).transform()
    print(wav_to_features.shape)
