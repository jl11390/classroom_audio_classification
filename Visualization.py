import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import public_func as f
import numpy as np
import os
from librosa.sequence import viterbi_binary


class Visualizer:
    def __init__(self, label_dict, reverse_label_dict):
        self.label_dict = label_dict
        self.reverse_label_dict = reverse_label_dict
        self.color_dict = {
            0: 'red',
            1: 'blue',
            2: 'green',
            3: 'yellow',
            4: 'orange',
            5: 'black',
            6: 'brown'}
        self.fig, self.axes = plt.subplots(3, 1, figsize=(18, 12))

    def _legend_without_duplicate_labels(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper right', bbox_to_anchor =(1,-0.1))

    def _save_plot(self, save_path, file_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.fig.savefig(os.path.join(save_path, file_name.replace('wav', 'png')))

    def _wav_annot(self, file_name, audio_path, reference_dict, estimated_dict):
        wav, sr = librosa.load(os.path.join(audio_path, file_name), sr=22050)

        ax = self.axes[2]
        librosa.display.waveshow(wav, sr, color='black', x_axis='s', ax=ax)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, len(wav) / sr)
        bottom, top = ax.get_ylim()

        ax.annotate('reference labels', xy=(0.5, bottom-0.1), annotation_clip=False)
        for frac_dict in reference_dict:
            if frac_dict['labels'][0].lower() != 'other':
                ax.axvspan(frac_dict['start'], frac_dict['end'], ymin=0, ymax=0.1,
                           color=self.color_dict[self.label_dict[frac_dict['labels'][0]]], label=frac_dict['labels'][0],
                           alpha=0.5, zorder=-100)

        ax.annotate('estimated labels', xy=(0.5, top+0.05), annotation_clip=False)
        ax.text(0, 0, 'estimation')
        for pred_dict in estimated_dict:
            ax.axvspan(pred_dict['event_onset'], pred_dict['event_offset'], ymin=0.9, ymax=1,
                       color=self.color_dict[self.label_dict[pred_dict['event_label']]], label=pred_dict['event_label'],
                       alpha=0.5, zorder=-100)

        self._legend_without_duplicate_labels(ax)

    def _binary_hm(self, binary_pred):
        sns.heatmap(binary_pred, yticklabels=[self.reverse_label_dict[i] for i in range(len(binary_pred))], cbar=False,
                    ax=self.axes[1])
        self.axes[1].set_xlabel('Number of splitted audio')

    def _prob_hm(self, proba_pred):
        prob = np.array([y_pred_prob_label[:, 1] for y_pred_prob_label in proba_pred])
        sns.heatmap(prob, yticklabels=[self.reverse_label_dict[i] for i in range(len(prob))], cbar=True, vmin=0, vmax=1,
                    cmap = 'coolwarm', ax=self.axes[0], center = 0.5, cbar_kws = dict(use_gridspec=False, location="top", ticks=list([i/10 for i in range(11)]), label = 'predicted probability'))
        self.axes[0].set_xlabel('Number of splitted audio')

    def plot(self, file_name, audio_path, save_path, reference_dict, estimated_dict, proba_pred, binary_pred):
        self._wav_annot(file_name, audio_path, reference_dict, estimated_dict)
        self._binary_hm(binary_pred)
        self._prob_hm(proba_pred)
        self._save_plot(save_path, file_name)


if __name__ == "__main__":
    label_dict = f.get_label_dict(0)
    reverse_label_dict = f.get_reverse_label_dict(0)
    file_name = '0e8afbf2-Film_1_005.wav'
    audio_path = 'data/COAS_2/Audios_test'
    save_path = 'data/COAS_2/Plots'
    with open('data/COAS_2/Features/proba_test.pkl', 'rb') as f:
        proba_pred = pickle.load(f)
    prob = np.array(
        [y_pred_prob_label[:, 1] for y_pred_prob_label in proba_pred])  # proba matrix [num classes, num samples]
    transit_prob = 0.5
    transition_mtx = np.array([[1 - transit_prob, transit_prob], [transit_prob, 1 - transit_prob]])
    num_label = prob.shape[0]
    transition_mtx_full = np.repeat(transition_mtx[np.newaxis, :, :], num_label, axis=0)
    binary_pred = viterbi_binary(prob, transition_mtx_full)

    reference_dict = [
        {
            "start": 0,
            "end": 599.0818069301298,
            "labels": [
                "Q/A"
            ]
        }
    ]
    estimated_dict = [{'event_onset': 0, 'event_offset': 83, 'event_label': 'Lecturing'},
                      {'event_onset': 82, 'event_offset': 133, 'event_label': 'Lecturing'},
                      {'event_onset': 132, 'event_offset': 153, 'event_label': 'Lecturing'},
                      {'event_onset': 152, 'event_offset': 161, 'event_label': 'Lecturing'},
                      {'event_onset': 160, 'event_offset': 173, 'event_label': 'Lecturing'},
                      {'event_onset': 174, 'event_offset': 193, 'event_label': 'Lecturing'},
                      {'event_onset': 276, 'event_offset': 281, 'event_label': 'Lecturing'},
                      {'event_onset': 378, 'event_offset': 383, 'event_label': 'Lecturing'},
                      {'event_onset': 446, 'event_offset': 451, 'event_label': 'Lecturing'},
                      {'event_onset': 130, 'event_offset': 135, 'event_label': 'Q/A'},
                      {'event_onset': 200, 'event_offset': 205, 'event_label': 'Q/A'},
                      {'event_onset': 226, 'event_offset': 231, 'event_label': 'Q/A'},
                      {'event_onset': 242, 'event_offset': 247, 'event_label': 'Q/A'},
                      {'event_onset': 264, 'event_offset': 269, 'event_label': 'Q/A'},
                      {'event_onset': 282, 'event_offset': 291, 'event_label': 'Q/A'},
                      {'event_onset': 300, 'event_offset': 305, 'event_label': 'Q/A'},
                      {'event_onset': 324, 'event_offset': 329, 'event_label': 'Q/A'},
                      {'event_onset': 328, 'event_offset': 335, 'event_label': 'Q/A'},
                      {'event_onset': 364, 'event_offset': 369, 'event_label': 'Q/A'},
                      {'event_onset': 368, 'event_offset': 375, 'event_label': 'Q/A'},
                      {'event_onset': 442, 'event_offset': 447, 'event_label': 'Q/A'},
                      {'event_onset': 454, 'event_offset': 461, 'event_label': 'Q/A'},
                      {'event_onset': 464, 'event_offset': 469, 'event_label': 'Q/A'},
                      {'event_onset': 476, 'event_offset': 481, 'event_label': 'Q/A'},
                      {'event_onset': 480, 'event_offset': 487, 'event_label': 'Q/A'},
                      {'event_onset': 486, 'event_offset': 493, 'event_label': 'Q/A'},
                      {'event_onset': 492, 'event_offset': 505, 'event_label': 'Q/A'},
                      {'event_onset': 518, 'event_offset': 523, 'event_label': 'Q/A'},
                      {'event_onset': 530, 'event_offset': 537, 'event_label': 'Q/A'},
                      {'event_onset': 550, 'event_offset': 555, 'event_label': 'Q/A'},
                      {'event_onset': 576, 'event_offset': 591, 'event_label': 'Q/A'},
                      {'event_onset': 592, 'event_offset': 597, 'event_label': 'Q/A'}]

    visualizer = Visualizer(label_dict, reverse_label_dict)
    visualizer.plot(file_name, audio_path, save_path, reference_dict, estimated_dict, proba_pred, binary_pred)
