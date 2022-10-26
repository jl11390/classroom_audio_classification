"""Classroom Activity Sound Effect Detection"""
import json
import os
import warnings
import pickle
import numpy as np
from DataLoader import DataLoader
from WavToFeatures import WavToFeatures
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import RandomizedSearchCV
from librosa.sequence import viterbi_binary
from WavToFeatures import WavToFeatures
from Evaluation import Evaluation
from Visualization import Visualizer
import public_func as f


class CASED:
    def __init__(self, frac_t, long_frac_t, step_t, target_class_version=0):
        # init info
        self.frac_t = frac_t
        self.step_t = step_t
        self.long_frac_t = long_frac_t
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

        self.label_dict = f.get_label_dict(target_class_version)
        self.reverse_label_dict = f.get_reverse_label_dict(target_class_version)

    def get_metadict(self, annot, file_name):
        for metadict in annot:
            # note: this way to get file name is not ideal
            file_name_mp4 = metadict['video_url'].split('/')[-1]
            file_name_wav = file_name_mp4.replace('.mp4', '.wav')
            if file_name == file_name_wav:
                return metadict
        return None

    def load_train_data(self, annot_path, audio_path, cache_path, load_cache=False, num_folds=5):
        """
        load all training data, using DataLoader, into self.features_matrix_all and self.labels_matrix_all and self.folds_all
        """
        with open(annot_path, 'r') as f:
            annot = json.load(f)
        audiofiles = [f for f in os.listdir(audio_path) if f.endswith('wav')]
        n = len(audiofiles)

        features_matrix_all = None
        labels_matrix_all = None
        folds_all = []

        for i, file_name in enumerate(audiofiles):
            metadict = self.get_metadict(annot, file_name)
            if metadict is None:
                continue
            dataloader = DataLoader(file_name, audio_path, cache_path, metadict, self.frac_t, self.long_frac_t,
                                    self.step_t, target_class_version=self.target_class_version)
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
            print(f'best params: {random_search.best_params_}')
            print(f'best score: {random_search.best_score_}')

            # update the best model
            self.best_model = random_search.best_estimator_

            # save the best model
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            with open(os.path.join(cache_path, 'best_estimator.pkl'), "wb") as f:
                pickle.dump(self.best_model, f)

    def predict_proba(self, file_name, audio_path, load_cache=False):
        assert self.best_model is not None, 'get the best model first!'
        features_matrix = WavToFeatures(file_name, audio_path, cache_path, self.frac_t, self.long_frac_t, self.step_t).transform(load_cache=load_cache)
        features_matrix = self.standard_scaler.fit_transform(features_matrix)
        y_pred_prob = self.best_model.predict_proba(features_matrix)

        return y_pred_prob

    def predict_binary(self, file_name, audio_path, transit_prob=0.05, load_cache=False):
        y_pred_prob = self.predict_proba(file_name, audio_path, load_cache=load_cache)

        prob = np.array(
            [y_pred_prob_label[:, 1] for y_pred_prob_label in y_pred_prob])  # proba matrix [num classes, num samples]
        transition_mtx = np.array([[1 - transit_prob, transit_prob], [transit_prob, 1 - transit_prob]])
        num_label = prob.shape[0]
        transition_mtx_full = np.repeat(transition_mtx[np.newaxis, :, :], num_label, axis=0)
        binary_pred = viterbi_binary(prob, transition_mtx_full)
        
        return binary_pred

    def predict_annotation(self, file_name, audio_path, transit_prob=0.05, load_cache=False):
        """predict the smoothed (onset,offset) sequence for each target class"""
        assert self.best_model is not None, 'get the best model first!'

        binary_pred = self.predict_binary(file_name, audio_path, transit_prob=transit_prob, load_cache=False)

        # Get start time, end time of consecutive 1s for each class
        append1 = np.zeros((binary_pred.shape[0], 1), dtype=int)
        counts_ext = np.column_stack((append1, binary_pred, append1))
        diffs = np.diff((counts_ext == 1).astype(int), axis=1)
        starts = np.argwhere(diffs == 1)
        stops = np.argwhere(diffs == -1)
        start_stop = np.column_stack((starts[:, 0], starts[:, 1], stops[:, 1] - 1))

        # Return (onset,offset) sequence for all target classes
        estimated_event_list = []
        inv_map = {v: k for k, v in self.label_dict.items()}
        for detected in start_stop:
            start_t = detected[1] * self.step_t
            end_t = detected[2] * self.step_t + self.frac_t
            estimated_event_list.append(
                {'event_onset': start_t, 'event_offset': end_t, 'event_label': inv_map[detected[0]]})
        return estimated_event_list

    def visualize_pred(self, file_name, audio_path, save_path, annot_path, transit_prob=0.5, load_cache=False):
        proba_pred = self.predict_proba(file_name, audio_path, load_cache=load_cache)
        binary_pred = self.predict_binary(file_name, audio_path, transit_prob=transit_prob, load_cache=load_cache)

        with open(annot_path, 'r') as f:
            annot = json.load(f)
        metadict = self.get_metadict(annot, file_name)
        reference_event_list = metadict['tricks']
        estimated_event_list = self.predict_annotation(file_name, audio_path, transit_prob=transit_prob, load_cache=load_cache)
        visualizer = Visualizer(self.label_dict, self.reverse_label_dict)
        visualizer.plot(file_name, audio_path, save_path, reference_event_list, estimated_event_list, proba_pred, binary_pred)


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    frac_t, long_frac_t, step_t = 5, 20, 2
    annot_path = 'data/COAS_2/Annotation/project-3-at-2022-10-16-23-25-0c5736a4.json'
    audio_path = 'data/COAS_2/Audios'
    cache_path = 'data/COAS_2/Features'
    model_cache_path = 'data/COAS_2/Model'
    audio_test_path = 'data/COAS_2/Audios_test'
    eval_result_path = 'data/COAS_2/Eval_test'
    plot_path = 'data/COAS_2/Plots'
    cased = CASED(frac_t, long_frac_t, step_t, target_class_version=0)
    cased.load_train_data(annot_path, audio_path, cache_path, load_cache=True, num_folds=5)
    cased.randomized_search_cv(n_iter_search=30, cache_path=model_cache_path, load_cache=True)

    # evaluate on test audios
    ##################
    # TODOs:
    # 1. add global feature of max 
    # 2. visualize prediction and evaluation result, for both train and test, for pred_proba and viterbi
    # 3. try class weight for RF
    # 4. sampling strategy and data augmentation
    # 5. sed_eval agg metrics
    # 6. remove 'Other' label

    audiofiles_test = [f for f in os.listdir(audio_test_path) if f.endswith('wav')]
    for test_audio in audiofiles_test:
        cased.visualize_pred(test_audio, audio_test_path, plot_path, annot_path, transit_prob=0.5, load_cache=True)