import pandas as pd
from pathlib import Path
from feature_extract import AudioFeature
from audio_splitter import AudioSplitter
from kfold_model import KfoldModel
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import numpy as np
import json

'''
Use this function to get metadata when ready
'''


def save_features(features, fn):
    output_path = 'data/COAS/Features'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = f"{output_path}/{fn}.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    # specify whether to use previous saved features and whether to overwrite
    load_feature = False

    with open('data/COAS/Annotation/project-3-at-2022-10-10-17-01-baad4ee5.json', 'r') as f:
        data = json.load(f)

    # specify time of fraction and step
    frac_t, step_t = 10, 2
    # use sampled data to build pipeline
    sampled_data = data[:10]
    num_audios = len(sampled_data)
    # list to hold all instances of audios
    audio_features_all = []
    for i, metadict in enumerate(sampled_data):
        file_name = metadict["video_url"].split("-")[-1]
        audio_features = []

        fn = file_name.replace(".mp4", "")
        transformed_path = f"data/COAS/Features/{fn}.pkl"
        file_path = f"data/COAS/Audios/{fn}.wav"

        if load_feature and os.path.isfile(transformed_path):
            print(f'reading features from {transformed_path}')
            # if the file exists as a .pkl already, then load it
            with open(transformed_path, "rb") as f:
                audio_features = pickle.load(f)
        else:
            print(f'extracting features from {file_path}')
            # if the file doesn't exist, then extract its features from the source data and save the result
            audiosplitter = AudioSplitter(file_path, metadict)
            audiosplitter.split_audio(frac_t, step_t)
            datas, labels, sr = audiosplitter.datas, audiosplitter.labels, audiosplitter.sr
            for data, label in zip(datas, labels):
                # assign fold according to the total number of audios
                fold = int((i + 1) * 10 / num_audios)
                audio = AudioFeature(data, sr, label, fold)
                audio.extract_features(["mfcc", "spectral", "rms"])
                audio_features.append(audio)

            save_features(audio_features, fn)

        audio_features_all.extend(audio_features)
        print(f'number of sub audios extracted: {len(audio_features_all)}')

    feature_matrix = np.vstack([audio.features for audio in audio_features_all])
    labels = np.array([audio.label for audio in audio_features_all])
    folds = np.array([audio.fold for audio in audio_features_all])

    print(f'final feature matrix shape: {feature_matrix.shape}')
    print(f'final label array shape:{labels.shape}')
    print(f'final fold array shape:{folds.shape}')
    print(f'number of folds: {np.unique(folds)}')

    # train model when data is ready
    model_cfg = dict(
        model=RandomForestClassifier(
            random_state=42,
            n_jobs=10,
            class_weight="balanced",
            n_estimators=500,
            bootstrap=True,
        ),
    )
    model = KfoldModel(feature_matrix, labels, folds, model_cfg)
    fold_acc = model.train_kfold()
    print(fold_acc)
