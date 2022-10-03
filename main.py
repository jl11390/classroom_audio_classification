import pandas as pd
from pathlib import Path
from feature_extract import AudioFeature
from audio_splitter import AudioSplitter
from kfold_model import KfoldModel
import os
import pickle
import numpy as np

'''
Use this function to get metadata when ready
'''
def save_features(features, file_path):
        out_name = file_path.split("/")[-1]
        out_name = out_name.replace(".wav", "")

        filename = f"data/{out_name}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(features, f)

if __name__ == "__main__":
    load_feature =  False

    # metadata = parse_metadata("data/label_demo.csv")
    meta_df = pd.DataFrame(data={'file_name':['test.wav'], 'metadata_file':['test.json'], 'fold':[1]})
    metadata = zip(meta_df["file_name"], meta_df["metadata_file"], meta_df["fold"])

    # specify time of fraction and step
    frac_t, step_t = 5, 1

    # list to hold all instances of audios
    audio_features_all = []
    for row in metadata:

        audio_features = []
        src_path, metadata_path, fold = row
        print(f'extracting features from {src_path}')
        fn = src_path.replace(".wav", "")
        transformed_path = f"data//{fn}.pkl"
        file_path = f"data/{src_path}"
        metadata_path = f"data//{fn}.json"
        audiosplitter = AudioSplitter(file_path, metadata_path, fold)

        if load_feature and os.path.isfile(transformed_path):
            # if the file exists as a .pkl already, then load it
            with open(transformed_path, "rb") as f:
                audio_features = pickle.load(f)
        else:
            # if the file doesn't exist, then extract its features from the source data and save the result
            

            audiosplitter.split_audio(frac_t, step_t)
            datas, labels, sr = audiosplitter.datas, audiosplitter.labels, audiosplitter.sr
            for data, label in zip(datas, labels):
                audio = AudioFeature(data, sr, label, fold)
                audio.extract_features(["mfcc", "spectral", "rms"])
                audio_features.append(audio)
        
        audio_features_all.extend(audio_features)

    save_features(audio_features_all, file_path)

    feature_matrix = np.vstack([audio.features for audio in audio_features_all])
    labels = np.array([audio.label for audio in audio_features_all])
    folds = np.array([audio.fold for audio in audio_features_all])

    print(f'final feature matrix shape: {feature_matrix.shape}')
    print(f'final label array shape:{labels.shape}')
    print(f'final fold array shape:{folds.shape}')

    # train model when data is ready
    '''
    model_cfg = dict(
        model=RandomForestClassifier(
            random_state=42,
            n_jobs=10,
            class_weight="balanced",
            n_estimators=500,
            bootstrap=True,
        ),
    )
    model = Model(feature_matrix, labels, folds, model_cfg)
    fold_acc = model.train_kfold()
    '''