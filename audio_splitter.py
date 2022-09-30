from cProfile import label
from dataclasses import dataclass
import librosa
import numpy as np
from pathlib import Path
import os

'''
this class should cut audio with respect to specified extend time and target time
'''
class AudioSplitter:
    def __init__(self, extend_t, target_t):
        """
        @extent_t: time for which left and right to the target time that we also include for feature extraction
        @target_t: target time length for which label we are learning and predicting
        splitted audio time length = target_t + 2 * extend_t
        """
        self.extend_t = extend_t
        self.target_t = target_t

    def split_audio(self, src_path, label_lst, time_lst):
        """
        cut audio into training samples with time length = target_t + 2 * extend_t
        @src_path: path to the audio file
        @label_lst: list of a labels of audio
        @time_lst: list of a+1 time intervals for labels of audio
        """
        assert len(label_lst) + 1 == len(time_lst)
        y, sr = librosa.load(src_path, mono=True, sr=None)

        # number of output data
        target_sample = self.target_t * sr
        extend_sample = self.extend_t * sr
        all_sample = len(y)
        n = int((all_sample - 2*extend_sample)/(target_sample))
        # init output labels and output data
        labels = np.zeros(n)
        datas = np.zeros(shape=(n, target_sample+2*extend_sample))

        # only take sub_audio with full extended length, which means the first extend_t and last extend_t audio labels are omitted for now
        for i in range(n):
            left_i = i*target_sample
            right_i = (i+1)*target_sample + 2*extend_sample
            datas[i] = y[left_i:right_i]
            
            target_left_t = i*self.target_t + self.extend_t
            labels[i] = label_lst[np.argmin(np.array(time_lst)<target_left_t)-1]

        return datas, labels, sr


if __name__ == "__main__":
    src_path = f"data/test.wav"
    label_lst, time_lst = [1, 1, 2, 3, 2, 3, 4], [0, 20, 55, 70, 150, 180, 220, 299.36572916666665]
    extend_t, target_t = 10, 1

    audiosplitter = AudioSplitter(extend_t, target_t)
    datas, labels, sr = audiosplitter.split_audio(src_path, label_lst, time_lst)
    print(datas.shape, labels.shape, sr)