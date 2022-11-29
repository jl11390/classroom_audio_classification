import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import public_func as F
import numpy as np
import os
from WavToFeatures import WavToFeatures
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import json
import warnings


class Visualizer:
    def __init__(self, file_name, audio_path, annot_path, cache_path, frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0):
        # init info
        self.frac_t = frac_t
        self.step_t = step_t
        self.long_frac_t = long_frac_t
        self.long_long_frac_t = long_long_frac_t
        self.target_class_version = target_class_version
        self.file_name, self.audio_path, self.annot_path, self.cache_path = file_name, audio_path, annot_path, cache_path

        self.wav, self.sr = librosa.load(os.path.join(audio_path, file_name), sr=22050)
        with open(annot_path, 'r') as f:
            self.annot = json.load(f)
        self.metadict = self.get_metadict(file_name)
        self.label_dict = F.get_label_dict(target_class_version)
        self.reverse_label_dict = F.get_reverse_label_dict(target_class_version)
        self.color_dict = {
            0: 'red',
            1: 'blue',
            2: 'green',
            3: 'yellow',
            4: 'orange',
            5: 'black',
            6: 'brown'}
        
    def get_metadict(self, file_name):
        for metadict in self.annot:
            # note: this way to get file name is not ideal
            file_name_mp4 = metadict['video_url'].split('/')[-1]
            file_name_wav = file_name_mp4.replace('.mp4', '.wav')
            if file_name == file_name_wav:
                return metadict
        return None

    def _legend_without_duplicate_labels(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper right', bbox_to_anchor =(1,-0.1))

    def _save_plot(self, fig, save_path, file_name, suffix):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, file_name.replace('.wav', f'{suffix}.png')))

    def _wav_annot(self, ax):

        librosa.display.waveshow(y=self.wav, sr=self.sr, color='black', x_axis='s', ax=ax)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, len(self.wav) / self.sr)
        bottom, top = ax.get_ylim()

        ax.annotate('reference labels', xy=(0.5, bottom-0.1), annotation_clip=False)
        for frac_dict in self.metadict['tricks']:
            if frac_dict['labels'][0].lower() != 'other':
                ax.axvspan(frac_dict['start'], frac_dict['end'], ymin=0, ymax=0.1,
                           color=self.color_dict[self.label_dict[frac_dict['labels'][0]]], label=frac_dict['labels'][0],
                           alpha=0.5, zorder=-100)

        self._legend_without_duplicate_labels(ax)

    def _rms(self, ax):
        X = librosa.stft(self.wav)
        Y = librosa.amplitude_to_db(np.abs(X), ref=np.max)

        rms = librosa.feature.rms(S=Y)
        times = librosa.times_like(rms)
        ax.semilogy(times, rms[0], label='RMS Energy')
        ax.set_xlabel('Time (seconds)')
        ax.set(ylabel='RMS Energy', title='RMS')
        ax.set_xlim(0, len(self.wav) / self.sr)

    def _mfcc(self, fig, ax):
        mfccs = librosa.feature.mfcc(y=self.wav, sr=self.sr, n_mfcc=20)
        img = librosa.display.specshow(mfccs, x_axis='s', ax=ax)
        ax.set_xlabel('Time (seconds)')

        axins1 = inset_axes(
            ax,
            width="30%",  # width: 50% of parent_bbox width
            height="5%",  # height: 5%
            loc="upper right",
        )
        axins1.xaxis.set_ticks_position("bottom")
        fig.colorbar(img, cax=axins1, orientation="horizontal")
        ax.set(ylabel='MFCCs', title='MFCC')
        ax.set_xlim(0, len(self.wav) / self.sr)

    def _spectral_contrast(self, fig, ax):
        S = np.abs(librosa.stft(self.wav))
        contrast = librosa.feature.spectral_contrast(S=S, sr=self.sr, n_bands=6, fmin=60)

        img2 = librosa.display.specshow(contrast, x_axis='s', ax=ax)
        ax.set_xlabel('Time (seconds)')

        axins1 = inset_axes(
            ax,
            width="30%",  # width: 50% of parent_bbox width
            height="5%",  # height: 5%
            loc="upper right",
        )
        axins1.xaxis.set_ticks_position("bottom")
        fig.colorbar(img2, cax=axins1, orientation="horizontal")
        ax.set(ylabel='Frequency bands', title='Spectral contrast')
        ax.set_xlim(0, len(self.wav) / self.sr)

    def _sub_feature_matrix(self, ax, fm, ylabel, title):
        ax_in = inset_axes(
            ax,
            width="30%",  # width: 50% of parent_bbox width
            height="5%",  # height: 5%
            loc="upper right",
        )
        ax_in.xaxis.set_ticks_position("bottom")

        sns.heatmap(fm.T, cbar=True, cbar_ax=ax_in,
                    cmap = 'coolwarm', ax=ax, cbar_kws = dict(orientation="horizontal"))
        ax.set_xlabel('Number of splitted audio')
        ax.set(ylabel=ylabel, title=title)

    def _feature_matrix(self, ax_spectral, ax_mfcc, ax_rms, load_cache=True):
        features_matrix = WavToFeatures(self.file_name, self.audio_path, self.cache_path, self.frac_t, self.long_frac_t,
                                        self.long_long_frac_t, self.step_t).transform(load_cache=load_cache,
                                                                                      multi_scaling=False)
        # for training data, the pickle saves [feature_matrix, label_matrix]
        if isinstance(features_matrix, list):
            features_matrix = features_matrix[0]

        rms_fm = features_matrix[:, 54:-1]
        mfcc_fm = features_matrix[:, :40]
        spectral_fm = features_matrix[:, 40:54]

        self._sub_feature_matrix(ax_spectral, spectral_fm, 'mean and std of spectral contrast bands', 'Spectral Contrast')
        self._sub_feature_matrix(ax_mfcc, mfcc_fm, 'mean and std of MFCCs', 'MFCC')
        self._sub_feature_matrix(ax_rms, rms_fm, 'mean and std of RMS', 'RMS')


    def plot_raw_features(self, save_path):
        fig, axes = plt.subplots(4, 1, figsize=(18, 16))
        self._wav_annot(axes[3])
        self._rms(axes[2])
        self._mfcc(fig, axes[1])
        self._spectral_contrast(fig, axes[0])
        self._save_plot(fig, save_path, self.file_name, '_raw_features')

    def plot_features_matrix(self, save_path):
        fig, axes = plt.subplots(4, 1, figsize=(18, 16))
        self._feature_matrix(axes[0], axes[1], axes[2])
        self._wav_annot(axes[3])
        self._save_plot(fig, save_path, self.file_name, '_features_matrix')

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    frac_t, long_frac_t, long_long_frac_t, step_t = 5, 20, 60, 2

    file_name = '7249b16a-Innovation_2_000.wav'
    audio_path = 'data/COAS_2/Audios_test'
    save_path = 'data/COAS_2/Plots'
    cache_path = 'data/COAS_2/Features_test'
    annot_path = 'data/COAS_2/Annotation/project-3-at-2022-10-16-23-25-0c5736a4.json'

    visualizer = Visualizer(file_name, audio_path, annot_path, cache_path, frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)
    visualizer.plot_features_matrix(save_path)
    visualizer.plot_raw_features(save_path)
