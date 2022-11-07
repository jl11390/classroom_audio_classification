"""Classroom Activity Sound Event Detection"""
import json
import os
import warnings
import pickle
import numpy as np
import public_func as F
import sed_eval
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
from yellowbrick.model_selection import FeatureImportances


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

    def load_train_data(self, annot_path, audio_path, cache_path, cache_aug_path, aug_dict_path, audio_aug_path,
                        load_cache=False, num_folds=5):
        """
        load all training data, using DataLoader, into self.features_matrix_all and self.labels_matrix_all and self.folds_all
        annot_path: annotation json file
        audio_path:
        """
        with open(annot_path, 'r') as f:
            annot = json.load(f)
        with open(aug_dict_path, 'r') as json_file:
            aug_dict = json.load(json_file)

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
                                    self.long_long_frac_t, self.step_t, target_class_version=self.target_class_version)
            _extract_from_dataloader(dataloader, load_cache)
            # to load the corresponding augmented files
            aug_lst = aug_dict[file_name.replace('.wav', '')]
            num_aug_lst = len(aug_lst)
            for aug_file_name in aug_lst:
                dataloader = DataLoader(aug_file_name, audio_aug_path, cache_aug_path, metadict, self.frac_t,
                                        self.long_frac_t,
                                        self.long_long_frac_t, self.step_t,
                                        target_class_version=self.target_class_version)
                _extract_from_dataloader(dataloader, load_cache)
            fold = i % num_folds
            folds = num_samples * [fold]
            folds_all.extend(folds)
            print(f"loaded {i + 1} audios and its corresponding {num_aug_lst} augmented audios and {n - i - 1} to go\n")
        folds_all = np.array(folds_all)

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
        assert self.features_matrix_all is not None, 'load training data first!'

        model_path = os.path.join(cache_path, 'best_estimator.pkl')
        if load_cache and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.best_model = pickle.load(f)
            print(f'model loaded from cache path {model_path}')
        else:
            clf = RandomForestClassifier()
            class_weight_lst = self._customize_class_weights_candidates()
            param_dist = {"n_estimators": [200, 300],
                          "max_features": sp_randint(20, 50),
                          "max_depth": sp_randint(8, 20),
                          "criterion": ['entropy'],
                          "class_weight": class_weight_lst}

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
            with open(model_path, "wb") as f:
                pickle.dump(self.best_model, f)

    # def get_feature_importance(self):
    #     assert self.best_model, 'train model first'
    #     # visualize feature importance
    #     feature_visualizer = FeatureImportances(self.best_model, is_fitted=True, labels=F.get_features_names(), topn=10,
    #                                             stack=True)
    #     feature_visualizer.fit(self.features_matrix_all, self.labels_matrix_all)
    #     feature_plot_path = 'data/COAS_2/Plots'
    #     if not os.path.exists(feature_plot_path):
    #         os.makedirs(feature_plot_path)
    #     feature_visualizer.show(outpath=feature_plot_path)

    def predict_proba(self, file_name, audio_path, cache_test_path, load_cache=False):
        assert self.best_model is not None, 'get the best model first!'
        features_matrix = WavToFeatures(file_name, audio_path, cache_test_path, self.frac_t, self.long_frac_t,
                                        self.long_long_frac_t, self.step_t).transform(load_cache=load_cache)
        features_matrix = self.standard_scaler.transform(features_matrix)
        y_pred_prob = self.best_model.predict_proba(features_matrix)

        return y_pred_prob

    def predict_binary(self, file_name, audio_path, cache_test_path, transit_prob_01=0.5, trans_prob_10=0.5,
                       load_cache=False):
        y_pred_prob = self.predict_proba(file_name, audio_path, cache_test_path, load_cache=load_cache)

        prob = np.array(
            [y_pred_prob_label[:, 1] for y_pred_prob_label in y_pred_prob])  # proba matrix [num classes, num samples]
        transition_mtx = np.array([[1 - transit_prob_01, transit_prob_01], [trans_prob_10, 1 - trans_prob_10]])
        num_label = prob.shape[0]
        transition_mtx_full = np.repeat(transition_mtx[np.newaxis, :, :], num_label, axis=0)
        binary_pred = viterbi_binary(prob, transition_mtx_full)

        return binary_pred

    def predict_annotation(self, file_name, audio_path, cache_test_path, transit_prob_01=0.5, trans_prob_10=0.5,
                           load_cache=False):
        """predict the smoothed (onset,offset) sequence for each target class"""
        assert self.best_model is not None, 'get the best model first!'

        binary_pred = self.predict_binary(file_name, audio_path, cache_test_path, transit_prob_01=transit_prob_01,
                                          trans_prob_10=trans_prob_10,
                                          load_cache=load_cache)

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
            start_t = detected[1] * self.step_t
            end_t = detected[2] * self.step_t + self.frac_t
            estimated_event_list.append(
                {'event_onset': start_t, 'event_offset': end_t, 'event_label': inv_map[detected[0]]})
        return estimated_event_list

    def evaluate_all(self, annot_path, audio_test_path, cache_test_path, eval_result_path, transit_prob_01=0.5,
                     trans_prob_10=0.5, load_cache=True):

        with open(annot_path, 'r') as f:
            annot = json.load(f)
        audiofiles_test = [i for i in os.listdir(audio_test_path) if i.endswith('wav')]

        # Get used event labels
        reference_event_list_all = []
        estimated_event_list_all = []
        for file in audiofiles_test:
            metadict = self.get_metadict(annot, file)
            if metadict is None:
                print("test audio not found in annot")
                continue
            else:
                reference_event_list = metadict['tricks']
            estimated_event_list = self.predict_annotation(file, audio_test_path, cache_test_path,
                                                           transit_prob_01=transit_prob_01, trans_prob_10=trans_prob_10,
                                                           load_cache=load_cache)
            for est_event_dict in estimated_event_list:
                est_event_dict['event_label'] = self.label_dict[est_event_dict['event_label']]
            for event_dict in reference_event_list:
                if event_dict['labels'][0].lower() != 'other':
                    event_dict['event_onset'] = event_dict['start']
                    event_dict['event_offset'] = event_dict['end']
                    event_dict['event_label'] = self.label_dict[event_dict['labels'][0]]
                del event_dict['start']
                del event_dict['end']
                del event_dict['labels']
            reference_event_list_all.append(reference_event_list)
            estimated_event_list_all.append(estimated_event_list)

        reference_event_list_all = list(itertools.chain.from_iterable(reference_event_list_all))
        estimated_event_list_all = list(itertools.chain.from_iterable(estimated_event_list_all))
        event_labels = sed_eval.util.event_list.unique_event_labels(reference_event_list_all)

        # Create metrics classes, define parameters
        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=event_labels,
                                                                         time_resolution=1.0)
        segment_based_metrics.evaluate(reference_event_list=reference_event_list_all,
                                       estimated_event_list=estimated_event_list_all)
        # Get all metrices
        all_class_wise_metrics = segment_based_metrics.results_class_wise_metrics()
        # Filter metrices and change output format
        eval_result = {}
        for label_class, result in list(all_class_wise_metrics.items()):
            label = self.reverse_label_dict[label_class]
            eval_result[label] = {999: '999'}
            eval_result[label]['f_measure'] = result['f_measure']['f_measure']
            eval_result[label]['precision'] = result['f_measure']['precision']
            eval_result[label]['recall'] = result['f_measure']['recall']
            eval_result[label]['error_rate'] = result['error_rate']['error_rate']
            del eval_result[label][999]

        # Save result to eval_result_path
        if not os.path.exists(eval_result_path):
            os.makedirs(eval_result_path)
        path = os.path.join(eval_result_path, 'result_all.json')
        with open(path, "w") as outfile:
            json.dump(eval_result, outfile)

    def visualize_pred(self, file_name, audio_path, cache_test_path, save_path, annot_path, transit_prob_01=0.5,
                       trans_prob_10=0.5,
                       load_cache=False):
        proba_pred = self.predict_proba(file_name, audio_path, cache_test_path, load_cache=load_cache)
        binary_pred = self.predict_binary(file_name, audio_path, cache_test_path, transit_prob_01=transit_prob_01,
                                          trans_prob_10=trans_prob_10,
                                          load_cache=load_cache)

        with open(annot_path, 'r') as f:
            annot = json.load(f)
        metadict = self.get_metadict(annot, file_name)
        reference_event_list = metadict['tricks']
        estimated_event_list = self.predict_annotation(file_name, audio_path, cache_test_path,
                                                       transit_prob_01=transit_prob_01, trans_prob_10=trans_prob_10,
                                                       load_cache=load_cache)
        visualizer = Visualizer(self.label_dict, self.reverse_label_dict)
        visualizer.plot(file_name, audio_path, save_path, reference_event_list, estimated_event_list, proba_pred,
                        binary_pred)


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
    eval_result_path = 'data/COAS_2/Eval_test'
    plot_path = 'data/COAS_2/Plots'
    cased = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)
    cased.load_train_data(annot_path, audio_path, cache_path, cache_aug_path, aug_dict_path, audio_aug_path,
                          load_cache=True, num_folds=5)
    cased.randomized_search_cv(n_iter_search=10, cache_path=model_cache_path, load_cache=True)
    cased.evaluate_all(annot_path, audio_test_path, cache_test_path, eval_result_path, transit_prob_01=0.5,
                       trans_prob_10=0.3, load_cache=True)

    audiofiles_test = [f for f in os.listdir(audio_test_path) if f.endswith('wav')]
    for test_audio in audiofiles_test:
        cased.visualize_pred(test_audio, audio_test_path, cache_test_path, plot_path, annot_path, transit_prob_01=0.5,
                             trans_prob_10=0.3, load_cache=True)
