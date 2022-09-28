import pandas as pd
from pathlib import Path
from feature_extract import AudioFeature
from kfold_model import KfoldModel
import os
import pickle
import numpy as np

'''
Use this function to get metadata when ready
'''
def parse_metadata(path):
    meta_df = pd.read_csv(path)
    meta_df = meta_df[["sliced_file_name", "label"]]
    meta = zip(meta_df["sliced_file_name"], meta_df["label"])

    return meta

if __name__ == "__main__":
    load_feature =  False

    # metadata = parse_metadata("metadata/metadata.csv")
    metadata = zip(['test.wav'], ['label1'], [1])

    # list to hold all instances of audios
    audio_features = []
    for row in metadata:

        file_name, label, fold = row
        print(f'extracting features from {file_name}')
        fn = file_name.replace(".wav", "")
        transformed_path = f"data//{fn}.pkl"

        if load_feature and os.path.isfile(transformed_path):
            # if the file exists as a .pkl already, then load it
            with open(transformed_path, "rb") as f:
                audio = pickle.load(f)
                audio_features.append(audio)
        else:
            # if the file doesn't exist, then extract its features from the source data and save the result
            src_path = f"data/{file_name}"
            audio = AudioFeature(src_path, label, fold)
            audio.extract_features(["mfcc", "spectral", "rms"])
            audio_features.append(audio)

    feature_matrix = np.vstack([audio.features for audio in audio_features])
    labels = np.array([audio.label for audio in audio_features])
    folds = np.array([audio.fold for audio in audio_features])

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