import librosa
import numpy as np


def get_label_dict(target_class_version):
    assert target_class_version in [0, 1], "target class is invalid"
    label_dict = None
    if target_class_version == 0:  # full target class
        label_dict = {
            'Lecturing': 0,
            'Q/A': 1,
            'Teacher-led Conversation': 2,
            'Student Presentation': 3,
            'Individual Student Work': 4,
            'Collaborative Student Work': 5,
        }
    if target_class_version == 1:  # reduced target class
        label_dict = {
            'Lecturing': 0,
            'Q/A': 0,
            'Teacher-led Conversation': 0,
            'Student Presentation': 0,
            'Individual Student Work': 1,
            'Collaborative Student Work': 2,
        }

    label_dict_lower = {}
    label_dict_upper = {}

    for target, target_class in label_dict.items():
        if target != target.lower():
            label_dict_lower[target.lower()] = target_class
        if target != target.upper():
            label_dict_upper[target.upper()] = target_class
    label_dict.update(label_dict_lower)
    label_dict.update(label_dict_upper)

    return label_dict


def get_reverse_label_dict(target_class_version):
    assert target_class_version in [0, 1], "target class is invalid"
    label_dict = None
    if target_class_version == 0:  # full target class
        label_dict = {
            'Lecturing': 0,
            'Q/A': 1,
            'Teacher-led Conversation': 2,
            'Student Presentation': 3,
            'Individual Student Work': 4,
            'Collaborative Student Work': 5,
        }
    if target_class_version == 1:  # reduced target class
        label_dict = {
            'Lecturing': 0,
            'Q/A': 0,
            'Teacher-led Conversation': 0,
            'Student Presentation': 0,
            'Individual Student Work': 1,
            'Collaborative Student Work': 2,
        }

    keys = list(label_dict.keys())
    values = list(label_dict.values())
    reverse_label_dict = {}
    for value, key in zip(values, keys):
        if value in reverse_label_dict:
            reverse_label_dict[value] = reverse_label_dict[value] + ' & ' + key
        else:
            reverse_label_dict[value] = key

    return reverse_label_dict


# extract the local rms max, which will be used as a feature
def get_local_rms_max(y):
    # window and hop here is for calculating rms in a splitted audio
    X = librosa.stft(y)
    Y = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    rms = librosa.feature.rms(S=Y)
    return rms.max()


# get features names for feature importance visualization
def get_features_names():
    label = []
    mfcc_mean = ['mfcc_mean_' + str(i) for i in range(20)]
    mfcc_variance = ['mfcc_variance_' + str(i) for i in range(20)]
    spec_con_mean = ['spec_con_mean_' + str(i) for i in range(7)]
    spec_con_variance = ['spec_con_variance_' + str(i) for i in range(7)]
    rms_mean = ['rms_mean']
    rms_variance = ['rms_variance']
    [label.extend(i) for i in [mfcc_mean, mfcc_variance, spec_con_mean, spec_con_variance, rms_mean, rms_variance]]
    original_features = label.copy()
    [label.append(i + '_diff_1') for i in original_features]
    [label.append(i + '_diff_2') for i in original_features]
    label.append('rms_local_max')
    return label


if __name__ == "__main__":
    label_dict = get_label_dict(0)
    print(label_dict)
    reverse_label_dict = get_reverse_label_dict(0)
    print(reverse_label_dict)
