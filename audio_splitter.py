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
    def __init__(self, src_path, metadata_dict):
        self.src_path = src_path
        self.metadata_dict = metadata_dict
        self.y, self.sr = librosa.load(src_path, mono=True, sr=None)
        self.datas = None
        self.features = None
        self.labels = None
        self.label_dict = {
            'Lecturing'.lower():0, 
            'Q/A'.lower():1, 
            'Teacher-led Conversation'.lower():2, 
            'Individual Student Work'.lower():3,
            'Collaborative Student Work'.lower():4,
            'Student Presentation'.lower():5,
            'Other'.lower():6
            }

    def _get_label(self, right_t, left_t):
        assert left_t < right_t
        t = right_t - left_t

        label_arr = np.zeros(len(self.label_dict))
        annot_result = self.metadata_dict['tricks']

        for i in range(len(annot_result)):
            start_t = annot_result[i]['start']
            end_t = annot_result[i]['end']
            labels = annot_result[i]['labels']
            if right_t >= start_t and left_t <= end_t:
                for label in labels:
                    label_arr[self.label_dict[label.lower()]] = 1
                break
            if left_t <= start_t and right_t > start_t and right_t - start_t >= 0.25 * t:
                for label in labels:
                    label_arr[self.label_dict[label.lower()]] = 1
            if right_t >= end_t and left_t < end_t and end_t - left_t >= 0.25 * t:
                for label in labels:
                    label_arr[self.label_dict[label.lower()]] = 1

        return label_arr

    def split_audio(self, frac_t, step_t):
        """
        @frac_t: cut audio into training samples with time length = frac_t
        @step_t: move the window forward step_t
        """

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
            self.labels[i] = self._get_label(right_t, left_t)


if __name__ == "__main__":
    metadata_dict = {
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

    frac_t, step_t = 5, 1
    src_path = 'data/COAS/Audios/Technology_1_008.wav'

    audiosplitter = AudioSplitter(src_path, metadata_dict)
    audiosplitter.split_audio(frac_t, step_t)
    print(audiosplitter.datas.shape, audiosplitter.labels.shape, audiosplitter.sr)