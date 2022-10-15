"""Classroom Activity Sound Effect Detection"""
import json
import os
import warnings
import pickle
import numpy as np
from DataLoader import DataLoader, wav_transform
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import RandomizedSearchCV
from librosa.sequence import viterbi_binary


class CASED:
    def __init__(self, frac_t, step_t, target_class_version=0):
        # init info
        self.frac_t = frac_t
        self.step_t = step_t
        self.target_class_version = target_class_version
        # load training data
        self.features_matrix_all = None
        self.labels_matrix_all = None
        self.folds_all = None
        # standard scaler
        self.standard_scaler = StandardScaler(copy=True)
        # best model
        self.best_model = None
        # evaluate accuracy on training data
        self.val_fold_scores_ = []

    def load_train_data(self, annot_path, audio_path, cache_path, load_cache=False, num_folds=5):
        """
        load all training data, using DataLoader, into self.features_matrix_all and self.labels_matrix_all and self.folds_all
        """
        with open(annot_path, 'r') as f:
            annot = json.load(f)
        n = len(annot)

        features_matrix_all = None
        labels_matrix_all = None
        folds_all = []

        for i, metadict in enumerate(annot):
            file_name_mp4 = metadict['video_url'].split('-')[-1]
            file_name = file_name_mp4.replace('.mp4', '.wav')

            dataloader = DataLoader(file_name, audio_path, cache_path, metadict, self.frac_t, self.step_t,
                                    target_class_version=self.target_class_version)
            features_matrix, labels_matrix = dataloader.load_data(load_cache=load_cache)

            features_matrix_all = np.vstack(
                [features_matrix_all, features_matrix]) if features_matrix_all is not None else features_matrix
            labels_matrix_all = np.vstack(
                [labels_matrix_all, labels_matrix]) if labels_matrix_all is not None else labels_matrix

            fold = i % num_folds
            folds = labels_matrix.shape[0] * [fold]
            folds_all.extend(folds)
            print(f"loaded {i + 1} audios and {n - i - 1} to go\n")
        folds_all = np.array(folds_all)

        self.features_matrix_all = features_matrix_all
        # standardization
        self.features_matrix_all = self.standard_scaler.fit_transform(self.features_matrix_all)
        self.labels_matrix_all = labels_matrix_all
        self.folds_all = folds_all
        print(
            f'training data loaded successfully! \n feature matrix shape:{self.features_matrix_all.shape} \n label matrix shape:{self.labels_matrix_all.shape}')
        print(
            f'proportion of labels in each target class: {np.sum(labels_matrix_all, axis=0) / np.sum(labels_matrix_all)}')

    def randomized_search_cv(self, n_iter_search=10, cache_path='data/COAS/Model', load_cache=False):
        """leave one group out cross validation for performance evaluation and model selection"""
        assert self.features_matrix_all is not None, 'load training data first!'

        model_path = os.path.join(cache_path, 'best_estimator.pkl')
        if load_cache and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.best_model = pickle.load(f)
            print(f'model loaded from cache path {model_path}')
        else:
            clf = RandomForestClassifier()
            param_dist = {"n_estimators": [100, 200, 300, 400],
                          "max_features": sp_randint(10, 50),
                          "max_depth": sp_randint(2, 10),
                          "criterion": ['entropy', 'gini']}

            # make_scorer wraps score function for use in cv, 'micro' calculates metrics globally by counting TP,FP,TN,FN
            f_scorer = make_scorer(f1_score, average='micro')

            logo = LeaveOneGroupOut()

            # Note: refit will fit an estimator using the best found parameters on the whole dataset
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, refit=True,
                                               scoring=f_scorer, cv=logo, n_jobs=-1, verbose=3)

            # run randomized cross validation
            random_search.fit(self.features_matrix_all, self.labels_matrix_all, groups=self.folds_all)
            print(random_search.best_params_)
            print(random_search.best_score_)

            # update the best model
            self.best_model = random_search.best_estimator_

            # save the best model
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            with open(os.path.join(cache_path, 'best_estimator.pkl'), "wb") as f:
                pickle.dump(self.best_model, f)

    def evaluate_accuracy(self):
        """evaluate the accuracy for training data, just for curiosity"""
        assert self.features_matrix_all is not None and self.best_model is not None, 'not now!'
        print('it might take a while, be patient!')
        logo = LeaveOneGroupOut()
        for train_index, test_index in logo.split(self.features_matrix_all, self.labels_matrix_all, self.folds_all):
            X_train, X_test = self.features_matrix_all[train_index], self.features_matrix_all[test_index]
            y_train, y_test = self.labels_matrix_all[train_index], self.labels_matrix_all[test_index]

            clf = self.best_model
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            # In multilabel classification, this function computes subset accuracy:
            # the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
            fold_acc = accuracy_score(y_test, y_pred)
            self.val_fold_scores_.append(fold_acc)

    def predict_viterbi_binary(self, src_path, transit_prob=0.05):
        """predict the smoothed (onset,offset) sequence for each target class"""
        assert self.best_model is not None, 'get the best model first!'
        features_matrix = wav_transform(src_path, self.frac_t, self.step_t)  # transform wav file into feature matrix
        features_matrix = self.standard_scaler.fit_transform(features_matrix)
        y_pred_prob = self.best_model.predict_proba(features_matrix)
        y_pred = self.best_model.predict(features_matrix)
        prob = np.array(
            [y_pred_prob_label[:, 1] for y_pred_prob_label in y_pred_prob])  # proba matrix [num classes, num samples]
        transition_mtx = np.array([[1 - transit_prob, transit_prob], [transit_prob, 1 - transit_prob]])
        num_label = prob.shape[0]
        transition_mtx_full = np.repeat(transition_mtx[np.newaxis, :, :], num_label, axis=0)
        binary_pred = viterbi_binary(prob, transition_mtx_full)


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    frac_t, step_t = 10, 2
    annot_path = 'data/COAS/Annotation/project-3-at-2022-10-10-17-01-baad4ee5.json'
    audio_path = 'data/COAS/Audios'
    cache_path = 'data/COAS/Features'
    model_cache_path = 'data/COAS/Model'
    test_audio = 'data/COAS/TestAudios/Technology_1_008.wav'
    cased = CASED(frac_t, step_t, target_class_version=0)
    cased.load_train_data(annot_path, audio_path, cache_path, load_cache=False, num_folds=5)
    cased.randomized_search_cv(n_iter_search=20, cache_path=model_cache_path, load_cache=False)
    cased.evaluate_accuracy()
    print(cased.val_fold_scores_)
    cased.predict_viterbi_binary(test_audio, transit_prob=0.05)
