import librosa
import numpy as np
import public_func as F


class AudioSplitter:
    def __init__(self, src_path, metadata_dict, target_class_version=0):
        """
        src_path: path of audio file
        metadata_dict: annotation dictionary
        target_class_version: version of the target class, set to 0 if working with full target class, set to 1 if working with reduced target class
        """
        self.src_path = src_path
        self.metadata_dict = metadata_dict
        self.y, self.sr = librosa.load(src_path, sr=22050, mono=True)
        self.long_y = None
        self.long_long_y = None
        self.datas = None
        self.long_datas = None
        self.long_long_datas = None
        self.labels = None
        self.transition_indicator = None
        self.label_dict = F.get_label_dict(target_class_version)
        self.label_dict = dict((k.lower(), v) for k, v in self.label_dict.items())
        self.labels_length = len(set(self.label_dict.values()))

    def split_audio(self, frac_t, long_frac_t, long_long_frac_t, step_t, threshold=0.3):
        """
        frac_t: frame length in seconds
        step_t: hop length in seconds
        threshold:
        """
        num_samples = len(self.y)  # total number of samples
        audio_length = num_samples / self.sr  # audio length in seconds
        num_samples_frame = frac_t * self.sr  # number of samples in each frame

        n_frames = ((audio_length - frac_t) // step_t) + 1  # total number of frames that could be extracted
        n_frames = int(n_frames)

        # To get long "multi-scaling"
        long_audio_length_1 = (n_frames - 1) * step_t + long_frac_t
        long_num_samples_1 = int(np.ceil(long_audio_length_1 * self.sr))
        num_paddings_1 = long_num_samples_1 - num_samples
        long_num_samples_frame_1 = long_frac_t * self.sr
        self.long_y = np.pad(self.y, (0, num_paddings_1), 'mean')

        # To get long-long"multi-scaling"
        long_audio_length_2 = (n_frames - 1) * step_t + long_long_frac_t
        long_num_samples_2 = int(np.ceil(long_audio_length_2 * self.sr))
        num_paddings_2 = long_num_samples_2 - num_samples
        long_num_samples_frame_2 = long_long_frac_t * self.sr
        self.long_long_y = np.pad(self.y, (0, num_paddings_2), 'mean')

        self.datas = np.zeros((n_frames, num_samples_frame))  # contains the frames extracted from audio
        self.long_datas = np.zeros((n_frames, long_num_samples_frame_1))
        self.long_long_datas = np.zeros((n_frames, long_num_samples_frame_2))
        self.labels = np.zeros((n_frames, self.labels_length))  # contains the multi-label
        self.transition_indicator = np.zeros(n_frames)

        for i in range(n_frames):
            left_t = i * step_t
            right_t = i * step_t + frac_t
            long_right_t = i * step_t + long_frac_t
            long_long_right_t = i * step_t + long_long_frac_t
            self.datas[i] = self.y[left_t * self.sr:right_t * self.sr]
            self.long_datas[i] = self.long_y[left_t * self.sr:long_right_t * self.sr]
            self.long_long_datas[i] = self.long_long_y[left_t * self.sr:long_long_right_t * self.sr]
            label_arr, transition = self._get_label(left_t, right_t, threshold)
            self.labels[i] = label_arr
            self.transition_indicator[i] = int(transition)

    def _get_label(self, left_t, right_t, threshold=0.3):
        """
        left_t: start time of a frame
        right_t: end time of a frame
        threshold:
        Return:
            label_arr: array of the form [0,1,1,0]. [0,0,0,0] indicates no label
            transition: boolean
        """
        assert left_t < right_t
        label_arr = np.zeros(self.labels_length)
        frac_t = right_t - left_t
        overlap_lst = []
        transition = False
        annot_result = self.metadata_dict['tricks']
        for i in range(len(annot_result)):
            start_t = annot_result[i]['start']
            end_t = annot_result[i]['end']
            labels = annot_result[i]['labels']
            if labels[0].lower() == 'other':  # exclude 'other' from our data samples
                continue
            overlap = 0
            if end_t >= left_t >= start_t:
                if right_t <= end_t:
                    overlap = frac_t
                else:
                    overlap = end_t - left_t
            if end_t >= right_t >= start_t > left_t:
                overlap = right_t - start_t
            if left_t < start_t and right_t > end_t:  # unlikely case
                overlap = end_t - start_t

            overlap_lst.append(overlap)

            if overlap >= frac_t * threshold:
                label_arr[self.label_dict[labels[0].lower()]] = 1
        for j in range(len(overlap_lst) - 1):
            if overlap_lst[j] > 0 and overlap_lst[j + 1] > 0:
                transition = True
        return label_arr, transition

    def remove_noisy_data(self, remove_no_label_data=True, remove_transition=True):
        """
        remove the data without a label (i.e[0,0,0,0]) and/or the data is a transition
        """
        n_frames = self.labels.shape[0]
        if remove_no_label_data:
            idx1 = self.labels.sum(axis=1) > 0
        else:
            idx1 = np.ones(n_frames).astype(bool)
        if remove_transition:
            idx2 = self.transition_indicator == 0
        else:
            idx2 = np.ones(n_frames).astype(bool)
        idx = (idx1 & idx2)
        self.datas = self.datas[idx]
        self.long_datas = self.long_datas[idx]
        self.long_long_datas = self.long_long_datas[idx]
        self.labels = self.labels[idx]
        self.transition_indicator = self.transition_indicator[idx]


if __name__ == "__main__":
    metadata_dict = {
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

    frac_t, long_frac_t, long_long_frac_t, step_t = 5, 20, 60, 2
    src_path = 'data/COAS_2/Audios/48ad890d-ActiveLearning_6.wav'
    audiosplitter = AudioSplitter(src_path, metadata_dict, target_class_version=0)
    audiosplitter.split_audio(frac_t, long_frac_t, long_long_frac_t, step_t, threshold=0.3)
    audiosplitter.remove_noisy_data(remove_no_label_data=True, remove_transition=False)
    print(audiosplitter.datas.shape)
    print(audiosplitter.long_datas.shape)
    print(audiosplitter.long_long_datas.shape)
    print(audiosplitter.labels.shape)
