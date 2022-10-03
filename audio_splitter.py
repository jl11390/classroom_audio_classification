from cProfile import label
from dataclasses import dataclass
import librosa
import numpy as np
from pathlib import Path
import json
import os

'''
this class should cut audio with respect to specified extend time and target time
'''
class AudioSplitter:
    def __init__(self, src_path, metadata_path, fold):
        self.src_path = src_path
        self.metadata_path = metadata_path
        self.fold = fold
        self.y, self.sr = librosa.load(src_path, mono=True, sr=None)
        self.datas = None
        self.features = None
        self.labels = None
        self.label_dict = {'Q&A':0, 'Lecture':1}

    def _get_label(self, right_t, left_t, metadata):
        assert left_t < right_t
        t = right_t - left_t

        label_arr = np.zeros(len(self.label_dict))
        annot_result = metadata[0]['annotations'][0]['result']
        annot_result[0]['value']

        for i in range(len(annot_result)):
            start_t = annot_result[i]['value']['start']
            end_t = annot_result[i]['value']['end']
            labels = annot_result[i]['value']['labels']
            if right_t >= start_t and left_t <= end_t:
                label_arr[self.label_dict[labels[0]]] = 1
                break
            if left_t <= start_t and right_t > start_t and right_t - start_t >= 0.25 * t:
                label_arr[self.label_dict[labels[0]]] = 1
            if right_t >= end_t and left_t < end_t and end_t - left_t >= 0.25 * t:
                label_arr[self.label_dict[labels[0]]] = 1

        return label_arr

    def split_audio(self, frac_t, step_t):
        """
        @frac_t: cut audio into training samples with time length = frac_t
        @step_t: move the window forward step_t
        """
        f = open(self.metadata_path)
        # f = open('data/test.json')
        metadata = json.load(f)
        f.close()

        # number of fractions & init arrays
        n = round(len(self.y)/(step_t*self.sr))
        self.datas = np.zeros(shape=(n, frac_t*self.sr))
        self.labels = np.zeros(shape=(n, len(self.label_dict)))

        for i in range(n):
            left_t = i * step_t
            left_i = left_t * self.sr
            right_t = left_t + frac_t
            right_i = right_t * self.sr
            if right_i > len(self.y):
                right_i = len(self.y)
                left_i = right_i - frac_t * self.sr
                right_t = right_i/self.sr
                left_t = left_i/self.sr
            self.datas[i] = self.y[left_i:right_i]
            self.labels[i] = self._get_label(right_t, left_t, metadata)


if __name__ == "__main__":
    src_path = f"data/test.wav"
    metadata_path = f"data/test.json"
    fold = 1
    frac_t, step_t = 5, 1

    audiosplitter = AudioSplitter(src_path, metadata_path, fold)
    audiosplitter.split_audio(frac_t, step_t)
    print(audiosplitter.datas.shape, audiosplitter.labels.shape, audiosplitter.sr)