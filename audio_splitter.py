import librosa
import numpy as np
from pathlib import Path
import os

'''
this class should cut audio with hop_size = 1s into length = 10s with labels
'''
class AudioSplitter:
    def __init__(self, src_path, label_lst, start_lst, end_lst):
        self.src_path = src_path
        self.y, self.sr = librosa.load(src_path, mono=True, sr=None)
        self.label_lst = label_lst
        self.start_lst = start_lst
        self.end_lst = end_lst

    def split_audio_to(self, audio_output_path, metadata_output_path):
        """
        cut audio into training samples and store the information in a csv
        """
        # TODO

if __name__ == "__main__":
    src_path = f"data/test.wav"
    audio_output_path = ""
    metadata_output_path = ""
    label_lst, start_lst, end_lst = [], [], []
    audiosplitter = AudioSplitter(src_path, label_lst, start_lst, end_lst)
    audiosplitter.split_audio_to(audio_output_path, metadata_output_path)