from os import X_OK
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader
import pickle
from sklearn.ensemble import RandomForestClassifier
from librosa.sequence import viterbi_binary
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ss = StandardScaler(copy=True)

    def train(self, X_train, y_train):
        X_train = self.ss.fit_transform(X_train)

        clf = self.cfg["model"]
        clf.fit(X_train, y_train)

    def pred_viterbi_binary(self, X, transit_prob=0.05, num_label=4):
        clf = self.cfg["model"]
        X = self.ss.transform(X)
        y_pred_prob = clf.predict_proba(X)
        prob = np.array([y_pred_prob_label[:,1] for y_pred_prob_label in y_pred_prob])
        a = np.array([[1-transit_prob, transit_prob], [transit_prob, 1-transit_prob]])
        transition = np.repeat(a[np.newaxis, :, :], num_label, axis=0)
        binary_pred = viterbi_binary(prob, transition)
        
        return binary_pred

    def evaluate(self, X, metadict):
        # TODO: Use binary pred result to evaluate performance on each label
        binary_pred = self.pred_viterbi_binary(X)
        
        return binary_pred

if __name__ == "__main__":
    model_cfg = dict(
        model=RandomForestClassifier(
            random_state=42,
            n_jobs=10,
            class_weight="balanced",
            n_estimators=500,
            bootstrap=True,
        ),
    )
    frac_t, step_t = 10, 2
    annot_path = 'data/COAS/Annotation/project-3-at-2022-10-10-17-01-baad4ee5.json'
    audio_path = 'data/COAS/Audios'
    cache_path = 'data/COAS/Features'
    load_cache = True
    train_val_per = 0.9
    num_folds = 5
    with open(annot_path, 'r') as f:
        annot = json.load(f)
    f.close()
    n = len(annot)
    n_train_val = int(n*train_val_per)
    print(f"{n_train_val} audios feed into the training pipeline")
    features_matrix_all = None
    labels_matrix_all = None
    for i, metadict in enumerate(annot):
        if i < n_train_val:
            file_name_mp4 = metadict['video_url'].split('-')[-1]
            file_name = file_name_mp4.replace('.mp4', '.wav')
            dataloader = DataLoader(file_name, audio_path, cache_path, metadict, frac_t, step_t)
            features_matrix, labels_matrix = dataloader.load_data(load_cache = load_cache)
            features_matrix_all = np.vstack([features_matrix_all, features_matrix]) if features_matrix_all is not None else features_matrix
            labels_matrix_all = np.vstack([labels_matrix_all, labels_matrix]) if labels_matrix_all is not None else labels_matrix
            print(f"loaded {i+1} audios and {n_train_val-i-1} to go")
    print(f'proportion of labels: {np.sum(labels_matrix_all, axis=0)/np.sum(labels_matrix_all)}')

    model = Model(model_cfg)
    model.train(features_matrix_all, labels_matrix_all)
    evaluation_results_all = []
    for i, metadict in enumerate(annot):
        if i >= n_train_val:
            file_name_mp4 = metadict['video_url'].split('-')[-1]
            file_name = file_name_mp4.replace('.mp4', '.wav')
            dataloader = DataLoader(file_name, audio_path, cache_path, metadict, frac_t, step_t)
            features_matrix, labels_matrix = dataloader.load_data(load_cache = load_cache)
            evaluation_results = model.evaluate(features_matrix, metadict)
            evaluation_results_all.append(evaluation_results)

    for evaluation_results in evaluation_results_all:
        print(evaluation_results.shape)