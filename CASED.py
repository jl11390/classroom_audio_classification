"""Classroom Activity Sound Event Detection"""
import json
import os
import warnings
import pickle
import numpy as np
import public_func as F
import itertools
from DataLoader import DataLoader
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import RandomizedSearchCV
from librosa.sequence import viterbi_binary
from WavToFeatures import WavToFeatures
from Visualization import Visualizer
from Evaluation import Evaluation
from tqdm import tqdm


class CASED:
    def __init__(self, frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0):
        # init info
        self.frac_t = frac_t
        self.step_t = step_t
        self.long_frac_t = long_frac_t
        self.long_long_frac_t = long_long_frac_t
        self.target_class_version = target_class_version
        # load training data
        self.features_matrix_all = None
        self.labels_matrix_all = None
        self.folds_all = None
        # standard scaler
        self.standard_scaler = StandardScaler(copy=True)
        # best model
        self.best_model = None
        self.best_trans_prob_01, self.best_trans_prob_10, self.best_p_state_weight = None, None, None

        self.label_dict = F.get_label_dict(target_class_version)
        self.reverse_label_dict = F.get_reverse_label_dict(target_class_version)

    def get_metadict(self, annot, file_name):
        for metadict in annot:
            # note: this way to get file name is not ideal
            file_name_mp4 = metadict['video_url'].split('/')[-1]
            file_name_wav = file_name_mp4.replace('.mp4', '.wav')
            if file_name == file_name_wav:
                return metadict
        return None

    def load_train_data(self, annot_path, audio_path, cache_path, audio_aug_path=None, cache_aug_path=None,
                        aug_dict_path=None, load_cache=False, num_folds=5, multi_scaling=True):
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

        def _extract_from_dataloader(dataloader, load_cache):
            nonlocal features_matrix_all, labels_matrix_all, num_samples
            features_matrix, labels_matrix = dataloader.load_data(load_cache=load_cache)
            features_matrix_all = np.vstack(
                [features_matrix_all, features_matrix]) if features_matrix_all is not None else features_matrix
            labels_matrix_all = np.vstack(
                [labels_matrix_all, labels_matrix]) if labels_matrix_all is not None else labels_matrix
            num_samples += labels_matrix.shape[0]

        for i, file_name in enumerate(audiofiles):
            metadict = self.get_metadict(annot, file_name)
            if metadict is None:
                continue
            num_samples = 0
            dataloader = DataLoader(file_name, audio_path, cache_path, metadict, self.frac_t, self.long_frac_t,
                                    self.long_long_frac_t, self.step_t, self.target_class_version)
            _extract_from_dataloader(dataloader, load_cache)
            # load the corresponding augmented files if provided
            if audio_aug_path:
                assert cache_aug_path and aug_dict_path, 'please provide the corresponding cache path and augmented data dictionary'
                with open(aug_dict_path, 'r') as json_file:
                    aug_dict = json.load(json_file)
                aug_lst = aug_dict[file_name.replace('.wav', '')]
                num_aug_lst = len(aug_lst)
                for aug_file_name in aug_lst:
                    dataloader = DataLoader(aug_file_name, audio_aug_path, cache_aug_path, metadict, self.frac_t,
                                            self.long_frac_t, self.long_long_frac_t, self.step_t,
                                            target_class_version=self.target_class_version)
                    _extract_from_dataloader(dataloader, load_cache)
                print(f"loaded {num_aug_lst} augmented audios for {file_name}")
            fold = i % num_folds
            folds = num_samples * [fold]
            folds_all.extend(folds)
            print(f"loaded {i + 1} audios and {n - i - 1} to go\n")

        folds_all = np.array(folds_all)
        if not multi_scaling:
            original_data_dim = int((features_matrix_all.shape[1] - 1) / 3)
            fancy_index = [k for k in range(original_data_dim)]
            fancy_index.append(-1)
            features_matrix_all = features_matrix_all[:, fancy_index]

        self.features_matrix_all = features_matrix_all
        # standardization
        self.features_matrix_all = self.standard_scaler.fit_transform(self.features_matrix_all)
        self.labels_matrix_all = labels_matrix_all
        self.folds_all = folds_all
        print(
            f'training data loaded successfully! \n feature matrix shape:{self.features_matrix_all.shape} \n label matrix shape:{self.labels_matrix_all.shape}')
        print(
            f'proportion of labels in each target class: {np.sum(self.labels_matrix_all, axis=0) / np.sum(self.labels_matrix_all)}')

    def _customize_class_weights_candidates(self):
        """customize the class weight"""
        class_weight_lst = []

        class_weight_1 = [{0: 1, 1: 1.5}, {0: 1, 1: 2}, {0: 1, 1: 2}, {0: 1, 1: 2},
                          {0: 1, 1: 10}, {0: 1, 1: 10}]

        class_weight_2 = [{0: 1, 1: 1}, {0: 1, 1: 1.5}, {0: 1, 1: 2}, {0: 1, 1: 2},
                          {0: 1, 1: 5}, {0: 1, 1: 5}]

        class_weight_3 = [{0: 1, 1: 0.8}, {0: 1, 1: 1.2}, {0: 1, 1: 5}, {0: 1, 1: 5},
                          {0: 1, 1: 10}, {0: 1, 1: 10}]

        class_weight_lst.append(class_weight_1)  # customization 1
        class_weight_lst.append(class_weight_2)  # customization 2
        class_weight_lst.append(class_weight_3)  # customization 3
        return class_weight_lst

    def randomized_search_cv(self, n_iter_search=10, cache_path='data/COAS_2/Model', load_cache=False):
        """leave one group out cross validation for performance evaluation and model selection"""
        model_path = os.path.join(cache_path, 'best_estimator.pkl')
        if load_cache and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.best_model = pickle.load(f)
            print(f'model loaded from cache path {model_path}')
        else:
            assert self.features_matrix_all is not None, 'load training data first!'
            clf = RandomForestClassifier()
            class_weight_lst = self._customize_class_weights_candidates()
            param_dist = {"n_estimators": [200],
                          "max_features": sp_randint(20, 50),
                          "max_depth": sp_randint(8, 20),
                          "criterion": ['entropy'],
                          "class_weight": class_weight_lst}

            # make_scorer wraps score function for use in cv, 'micro' calculates metrics globally by counting TP,FP,TN,FN
            # use 'macro' instead!
            f_scorer = make_scorer(f1_score, average='macro')

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
            with open(model_path, "wb") as f:
                pickle.dump(self.best_model, f)

    def predict_proba(self, file_name, audio_path, cache_path, load_cache=False, multi_scaling=True):
        assert self.best_model is not None, 'get the best model first!'
        features_matrix = WavToFeatures(file_name, audio_path, cache_path, self.frac_t, self.long_frac_t,
                                        self.long_long_frac_t, self.step_t).transform(load_cache=load_cache,
                                                                                      multi_scaling=multi_scaling)

        # for training data, the pickle saves [feature_matrix, label_matrix]
        if isinstance(features_matrix, list):
            features_matrix = features_matrix[0]

        features_matrix = self.standard_scaler.transform(features_matrix)
        y_pred_prob = self.best_model.predict_proba(features_matrix)

        return y_pred_prob

    def predict_threshold(self, file_name, audio_path, cache_path, load_cache=False, multi_scaling=True):
        y_pred_prob = self.predict_proba(file_name, audio_path, cache_path, load_cache=load_cache,
                                         multi_scaling=multi_scaling)

        y_pred_threshold = np.array(
            [(y_pred_prob_label[:, 1] > 0.5).astype(int) for y_pred_prob_label in
             y_pred_prob])  # proba matrix [num classes, num samples]

        return y_pred_threshold

    def predict_viterbi(self, file_name, audio_path, cache_path, trans_prob_01=0.5, trans_prob_10=0.5,
                        p_state_weight=0.1, load_cache=False, multi_scaling=True):
        y_pred_prob = self.predict_proba(file_name, audio_path, cache_path, load_cache=load_cache,
                                         multi_scaling=multi_scaling)

        prob = np.array(
            [y_pred_prob_label[:, 1] for y_pred_prob_label in y_pred_prob])  # proba matrix [num classes, num samples]
        transition_mtx = np.array([[1 - trans_prob_01, trans_prob_01], [trans_prob_10, 1 - trans_prob_10]])
        num_label = prob.shape[0]
        transition_mtx_full = np.repeat(transition_mtx[np.newaxis, :, :], num_label, axis=0)
        # set p_state to be the proportion of state in the training data
        p_state = p_state_weight * (np.sum(self.labels_matrix_all, axis=0) / np.sum(self.labels_matrix_all)) + (
                1 - p_state_weight) * np.repeat(0.5, num_label)
        binary_pred = viterbi_binary(prob, transition_mtx_full, p_state=p_state)

        return binary_pred

    def predict_annotation(self, file_name, audio_path, cache_path, trans_prob_01=0.5, trans_prob_10=0.5,
                           p_state_weight=0.1, predict_type='viterbi_with_pstate',
                           load_cache=False, multi_scaling=True):
        """predict the smoothed (onset,offset) sequence for each target class"""
        assert self.best_model is not None, 'get the best model first!'
        assert predict_type in ['viterbi_with_pstate', 'viterbi_without_pstate', 'threshold']

        if predict_type == 'viterbi_with_pstate':
            binary_pred = self.predict_viterbi(file_name, audio_path, cache_path, trans_prob_01=trans_prob_01,
                                               trans_prob_10=trans_prob_10, p_state_weight=p_state_weight,
                                               load_cache=load_cache, multi_scaling=multi_scaling)
        elif predict_type == 'viterbi_without_pstate':
            binary_pred = self.predict_viterbi(file_name, audio_path, cache_path, trans_prob_01=trans_prob_01,
                                               trans_prob_10=trans_prob_10, p_state_weight=0,
                                               load_cache=load_cache, multi_scaling=multi_scaling)
        else:
            binary_pred = self.predict_threshold(file_name, audio_path, cache_path, load_cache=load_cache,
                                                 multi_scaling=multi_scaling)

        # Get start time, end time of consecutive 1s for each class
        append1 = np.zeros((binary_pred.shape[0], 1), dtype=int)
        counts_ext = np.column_stack((append1, binary_pred, append1))
        diffs = np.diff((counts_ext == 1).astype(int), axis=1)
        starts = np.argwhere(diffs == 1)
        stops = np.argwhere(diffs == -1)
        start_stop = np.column_stack((starts[:, 0], starts[:, 1], stops[:, 1] - 1))

        # Return (onset,offset) sequence for all target classes
        estimated_event_list = []
        inv_map = self.reverse_label_dict
        for detected in start_stop:
            if detected[0] not in self.reverse_label_dict.keys():
                continue
            start_t = detected[1] * self.step_t
            end_t = detected[2] * self.step_t + self.frac_t
            estimated_event_list.append(
                {'event_onset': start_t, 'event_offset': end_t, 'event_label': inv_map[detected[0]]})
        return estimated_event_list

    def evaluate_all(self, annot_path, audio_path, cache_path, eval_result_path, trans_prob_01=None, trans_prob_10=None,
                     p_state_weight=None, plot=False, predict_type='viterbi_with_pstate', load_cache=True,
                     multi_scaling=True):
        if trans_prob_01 is None and trans_prob_10 is None and p_state_weight is None:
            assert self.best_trans_prob_01 is not None, 'search viterbi first!'
            trans_prob_01, trans_prob_10, p_state_weight = self.best_trans_prob_01, self.best_trans_prob_10, self.best_p_state_weight

        with open(annot_path, 'r') as f:
            annot = json.load(f)
        audiofiles_test = [i for i in os.listdir(audio_path) if i.endswith('wav')]

        # Get used event labels
        reference_event_list_all = []
        estimated_event_list_all = []
        for file in audiofiles_test:
            metadict = self.get_metadict(annot, file)
            if metadict is None:
                print(f"evaluation audio {file} not found in annot")
                continue
            else:
                reference_event_list = metadict['tricks']
            estimated_event_list = self.predict_annotation(file, audio_path, cache_path,
                                                           trans_prob_01=trans_prob_01, trans_prob_10=trans_prob_10,
                                                           p_state_weight=p_state_weight, predict_type=predict_type,
                                                           load_cache=load_cache, multi_scaling=multi_scaling)
            for est_event_dict in estimated_event_list:
                est_event_dict['event_label'] = self.label_dict[est_event_dict['event_label']]
            for event_dict in reference_event_list:
                if event_dict['labels'][0].lower() != 'other':
                    event_dict['event_onset'] = event_dict['start']
                    event_dict['event_offset'] = event_dict['end']
                    event_dict['event_label'] = self.label_dict[event_dict['labels'][0].lower()]
                del event_dict['start']
                del event_dict['end']
                del event_dict['labels']
                if event_dict:
                    reference_event_list_all.append(event_dict)
            # reference_event_list_all.append(reference_event_list)
            estimated_event_list_all.append(estimated_event_list)

        # reference_event_list_all = list(itertools.chain.from_iterable(reference_event_list_all))
        estimated_event_list_all = list(itertools.chain.from_iterable(estimated_event_list_all))

        # print(estimated_event_list_all)
        evaluation = Evaluation(self.reverse_label_dict, 1.0, reference_event_list_all, estimated_event_list_all,
                                eval_result_path, target_class_version=self.target_class_version)
        eval_result, macro_f = evaluation.get_metrics()
        if plot:
            evaluation.plot_metrics(eval_result)
            confusion_matrix, event_labels = evaluation.get_confusion_matrix(evaluated_length_seconds=None)
            evaluation.plot_confusion_matrix(confusion_matrix, event_labels)
        print(f'evaluated audios in {audio_path}')

        return macro_f

    def search_viterbi_params(self, annot_path, audio_path, cache_path, params_cache_path, eval_result_path,
                              load_cache_params=False, multi_scaling=True):
        params_path = os.path.join(params_cache_path, 'best_params.pkl')
        if load_cache_params and os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                self.best_trans_prob_01, self.best_trans_prob_10, self.best_p_state_weight = pickle.load(f)
            print(f'params loaded from cache path {params_path}')
        else:
            trans_prob_01_list, trans_prob_10_list = np.arange(1, 10) / 10, np.arange(1, 10) / 10
            p_state_weight_list = np.arange(10) / 10
            params_arr = np.array(np.meshgrid(trans_prob_01_list, trans_prob_10_list, p_state_weight_list)).T.reshape(
                -1, 3)

            best_macro_f = -1
            for params in tqdm(params_arr):
                trans_prob_01, trans_prob_10, p_state_weight = params[0], params[1], params[2]
                macro_f = self.evaluate_all(annot_path, audio_path, cache_path, eval_result_path,
                                            trans_prob_01=trans_prob_01, trans_prob_10=trans_prob_10,
                                            p_state_weight=p_state_weight, plot=False, load_cache=True,
                                            multi_scaling=multi_scaling)
                if macro_f > best_macro_f:
                    best_macro_f = macro_f
                    self.best_trans_prob_01, self.best_trans_prob_10, self.best_p_state_weight = trans_prob_01, trans_prob_10, p_state_weight
            print(
                f'best transition proba 0-1: {self.best_trans_prob_01}, best transition proba 1-0: {self.best_trans_prob_10}, best p_state weight: {self.best_p_state_weight}')
            params_set = (self.best_trans_prob_01, self.best_trans_prob_10, self.best_p_state_weight)
            # save the best model

            if not os.path.exists(params_cache_path):
                os.makedirs(params_cache_path)
            with open(params_path, "wb") as f:
                pickle.dump(params_set, f)

    def visualize_pred(self, file_name, audio_path, cache_test_path, save_path, annot_path, trans_prob_01=None,
                       trans_prob_10=None, p_state_weight=None, predict_type='viterbi_with_pstate',
                       load_cache=False, multi_scaling=True):
        assert predict_type in ['viterbi_with_pstate', 'viterbi_without_pstate', 'threshold']
        if trans_prob_01 is None and trans_prob_10 is None and p_state_weight is None:
            assert self.best_trans_prob_01 is not None, 'search viterbi first!'
            trans_prob_01, trans_prob_10, p_state_weight = self.best_trans_prob_01, self.best_trans_prob_10, self.best_p_state_weight

        if predict_type == 'viterbi_with_pstate':
            binary_pred = self.predict_viterbi(file_name, audio_path, cache_path, trans_prob_01=trans_prob_01,
                                               trans_prob_10=trans_prob_10, p_state_weight=p_state_weight,
                                               load_cache=load_cache, multi_scaling=multi_scaling)
        elif predict_type == 'viterbi_without_pstate':
            binary_pred = self.predict_viterbi(file_name, audio_path, cache_path, trans_prob_01=trans_prob_01,
                                               trans_prob_10=trans_prob_10, p_state_weight=0,
                                               load_cache=load_cache, multi_scaling=multi_scaling)
        else:
            binary_pred = self.predict_threshold(file_name, audio_path, cache_path, load_cache=load_cache,
                                                 multi_scaling=multi_scaling)
        proba_pred = self.predict_proba(file_name, audio_path, cache_test_path, load_cache=load_cache,
                                        multi_scaling=multi_scaling)

        with open(annot_path, 'r') as f:
            annot = json.load(f)
        metadict = self.get_metadict(annot, file_name)
        reference_event_list = metadict['tricks']
        estimated_event_list = self.predict_annotation(file_name, audio_path, cache_test_path,
                                                       trans_prob_01=trans_prob_01, trans_prob_10=trans_prob_10,
                                                       predict_type=predict_type,
                                                       load_cache=load_cache, multi_scaling=multi_scaling)
        visualizer = Visualizer(self.label_dict, self.reverse_label_dict)
        visualizer.plot(file_name, audio_path, save_path, reference_event_list, estimated_event_list, proba_pred,
                        binary_pred)
        print(f'visualized audio in {file_name}')


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    frac_t, long_frac_t, long_long_frac_t, step_t = 5, 20, 60, 2

    annot_path = 'data/COAS_2/Annotation/project-3-at-2022-10-16-23-25-0c5736a4.json'
    audio_path = 'data/COAS_2/Audios'
    audio_aug_path = 'data/COAS_2/Audios_augmented'
    aug_dict_path = 'data/COAS_2/Annotation/file_aug_dict.json'
    cache_path = 'data/COAS_2/Features'
    cache_aug_path = 'data/COAS_2/Aug_features'
    cache_test_path = 'data/COAS_2/Features_test'
    model_cache_path = 'data/COAS_2/Model'
    audio_test_path = 'data/COAS_2/Audios_test'
    eval_test_path = 'data/COAS_2/Eval_test'
    eval_val_path = 'data/COAS_2/Eval_val'
    plot_path = 'Plots'

    cased = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased.load_train_data(annot_path, audio_path, cache_path, audio_aug_path, cache_aug_path, aug_dict_path,
                          load_cache=True, num_folds=5, multi_scaling=True)

    cased.randomized_search_cv(n_iter_search=10, cache_path=model_cache_path, load_cache=True)

    cased.search_viterbi_params(annot_path, audio_path, cache_path, model_cache_path, eval_val_path,
                                load_cache_params=True, multi_scaling=True)

    cased.evaluate_all(annot_path, audio_test_path, cache_test_path, eval_test_path, plot=True,
                       predict_type='threshold', load_cache=True, multi_scaling=True)

    audiofiles_test = [f for f in os.listdir(audio_test_path) if f.endswith('wav')]
    for test_audio in audiofiles_test:
        cased.visualize_pred(test_audio, audio_test_path, cache_test_path, plot_path, annot_path,
                             predict_type='viterbi_with_pstate', load_cache=True, multi_scaling=True)