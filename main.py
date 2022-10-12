from feature_extract import AudioFeature
from AudioSplitter import AudioSplitter
from kfold_model import KfoldModel
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import numpy as np
import json
import warnings


def load_and_train(annot_path, audio_path, cache_path, frac_t, step_t, load_cache=False, num_folds=5):
    with open(annot_path, 'r') as f:
        annot = json.load(f)
    n = len(annot)
    print(f"{n} audios feed into the pipeline")
    if load_cache:
        with open(cache_path, 'rb') as f:
            features_labels_folds = pickle.load(f)
        feature_matrix, labels_matrix, folds = features_labels_folds
    else:
        feature_matrix = None
        labels_matrix = None
        folds = []
        for i, metadict in enumerate(annot):
            file_name_mp4 = metadict['video_url'].split('-')[-1]
            file_name = file_name_mp4.replace('.mp4', '.wav')
            file_path = os.path.join(audio_path, file_name)
            audiosplitter = AudioSplitter(file_path, metadict, target_class_version=1)
            audiosplitter.split_audio(frac_t, step_t, threshold=0.3)
            audiosplitter.remove_noisy_data(remove_no_label_data=True, remove_transition=True)
            datas, labels = audiosplitter.datas, audiosplitter.labels
            for idx, bundle in enumerate(zip(datas, labels)):
                data, label = bundle
                fold = idx & num_folds
                audio_feature = AudioFeature(data, label, fold)
                audio_feature.extract_features(['mfcc', 'spectral', 'rms'])
                feature_matrix = np.vstack([feature_matrix, audio_feature.features]) if feature_matrix is not None else audio_feature.features
                labels_matrix = np.vstack([labels_matrix, audio_feature.label]) if labels_matrix is not None else audio_feature.label
                folds.append(fold)
            print(f"preprocessed {i+1} audios and {n-i-1} to go")
        folds = np.array(folds)
        features_labels_folds = [feature_matrix, labels_matrix, folds]
        save_pickle(features_labels_folds, cache_path)
    model_cfg = dict(
        model=RandomForestClassifier(
            random_state=42,
            n_jobs=10,
            class_weight="balanced",
            n_estimators=500,
            bootstrap=True,
        ),
    )
    model = KfoldModel(feature_matrix, labels_matrix, folds, model_cfg)
    fold_acc = model.train_kfold()
    print(fold_acc)


def save_pickle(features_labels_folds, cache_path):
    output_path = 'data/COAS/Features'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(cache_path, "wb") as f:
        pickle.dump(features_labels_folds, f)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    frac_t, step_t = 10, 2
    annot_path = 'data/COAS/Annotation/project-3-at-2022-10-10-17-01-baad4ee5.json'
    audio_path = 'data/COAS/Audios'
    cache_path = 'data/COAS/Features/preprocessing.pkl'
    load_and_train(annot_path, audio_path, cache_path, frac_t, step_t, load_cache=False, num_folds=5)
