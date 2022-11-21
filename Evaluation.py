import json
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import sed_eval
from sed_eval import util
import public_func as F


class Evaluation:
    def __init__(self, reverse_label_dict, time_resolution, reference_event_list_all, estimated_event_list_all, eval_result_path, target_class_version):
        self.reverse_label_dict = reverse_label_dict
        self.time_resolution = time_resolution
        self.reference_event_list_all = reference_event_list_all
        self.estimated_event_list_all = estimated_event_list_all
        self.eval_result_path = eval_result_path
        self.target_class_version = target_class_version
        self.event_labels = list(set(F.get_label_dict(target_class_version=self.target_class_version).values()))

    def get_metrics(self):
        # Create metrics classes, define parameters
        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=self.event_labels, time_resolution=1.0)
        segment_based_metrics.evaluate(reference_event_list=self.reference_event_list_all,
                                       estimated_event_list=self.estimated_event_list_all)
        # Get all metrices
        all_class_wise_metrics = segment_based_metrics.results_class_wise_metrics()
        # Filter metrices and change output format
        eval_result = {}
        for label_class, result in list(all_class_wise_metrics.items()):
            label = self.reverse_label_dict[label_class]
            eval_result[label] = {}
            eval_result[label]['f_measure'] = result['f_measure']['f_measure'] if np.isnan(result['f_measure']['f_measure']) == False else 0.0
            eval_result[label]['precision'] = result['f_measure']['precision'] if np.isnan(result['f_measure']['precision']) == False else 0.0
            eval_result[label]['recall'] = result['f_measure']['recall'] if np.isnan(result['f_measure']['recall']) == False else 0.0
            eval_result[label]['error_rate'] = result['error_rate']['error_rate'] if result['error_rate']['error_rate'] < 1 else 1

        # compute macro average of f1 score
        overall_f, c = 0, 1
        for label in eval_result:
            overall_f += eval_result[label]['f_measure']
            c += 1
        overall_f /= c

        # Save result to eval_result_path
        if not os.path.exists(self.eval_result_path):
            os.makedirs(self.eval_result_path)
        path = os.path.join(self.eval_result_path, 'result_all.json')
        with open(path, "w") as outfile:
            json.dump(eval_result, outfile)
        
        return eval_result, overall_f

    def plot_metrics(self, eval_result):
        df = pd.DataFrame.from_dict(eval_result).T
        df.plot(kind='bar', rot=15, figsize=(12,7))

        if not os.path.exists(self.eval_result_path):
            os.makedirs(self.eval_result_path)
        path = os.path.join(self.eval_result_path, 'metrics.png')
        plt.savefig(path)

    def get_confusion_matrix(self, evaluated_length_seconds=None):
        # Replicated from https://tut-arg.github.io/sed_eval/_modules/sed_eval/sound_event.html#SegmentBasedMetrics.evaluate

        # Convert event list into frame-based representation 
        reference_event_roll = util.event_list_to_event_roll(
            source_event_list=self.reference_event_list_all,
            event_label_list=self.event_labels,
            time_resolution=self.time_resolution
        )

        estimated_event_roll = util.event_list_to_event_roll(
            source_event_list=self.estimated_event_list_all,
            event_label_list=self.event_labels,
            time_resolution=self.time_resolution
        )

        max_offset = 0
        for item in self.reference_event_list_all:
            if item['event_offset'] > max_offset:
                max_offset = item['event_offset']
        for item in self.estimated_event_list_all:
            if item['event_offset'] > max_offset:
                max_offset = item['event_offset']

        if evaluated_length_seconds is None:
            evaluated_length_seconds = max_offset
            evaluated_length_segments = int(math.ceil(evaluated_length_seconds * 1 / float(self.time_resolution)))

        else:
            evaluated_length_segments = int(math.ceil(evaluated_length_seconds * 1 / float(self.time_resolution)))

        reference_event_roll, estimated_event_roll = util.match_event_roll_lengths(
            reference_event_roll,
            estimated_event_roll,
            evaluated_length_segments
        )

        reference_event_roll = np.array(reference_event_roll)
        estimated_event_roll = np.array(estimated_event_roll)
        
        confusion_matrix = multilabel_confusion_matrix(reference_event_roll, estimated_event_roll)

        labels = []
        for label in self.event_labels:
            labels.append(self.reverse_label_dict[label])

        return confusion_matrix, labels
    
    def plot_single_confusion_matrix(self, cfs_matrix, axes, class_label, class_names, fontsize=14):
        df_cm = pd.DataFrame(cfs_matrix, index=class_names, columns=class_names)
        
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title(class_label)
    
    def plot_confusion_matrix(self, confusion_matrix, event_labels):
        fig, ax = plt.subplots(2,3, figsize=(12,7))
        #ax[1][2].set_visible(False)
        for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, event_labels):
            self.plot_single_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
        
        fig.tight_layout()
        if not os.path.exists(self.eval_result_path):
            os.makedirs(self.eval_result_path)
        path = os.path.join(self.eval_result_path, 'confusion_matrix.png')
        plt.savefig(path)


if __name__ == "__main__":

    reverse_label_dict = {0: 'Lecturing', 1: 'Q/A', 2: 'Teacher-led Conversation', 3: 'Student Presentation', 4: 'Individual Student Work', 5: 'Collaborative Student Work'}

    reference_event_list_all = [
        {
            "event_onset": 0,
            "event_offset": 599.0818069301298,
            "event_label": 1
        }
    ]
    estimated_event_list_all = [
                      {'event_onset': 130, 'event_offset': 135, 'event_label': 1},
                      {'event_onset': 200, 'event_offset': 205, 'event_label': 1},
                      {'event_onset': 226, 'event_offset': 231, 'event_label': 1},
                      {'event_onset': 242, 'event_offset': 247, 'event_label': 1},
                      {'event_onset': 264, 'event_offset': 269, 'event_label': 1},
                      {'event_onset': 282, 'event_offset': 291, 'event_label': 1},
                      {'event_onset': 300, 'event_offset': 305, 'event_label': 1},
                      {'event_onset': 324, 'event_offset': 329, 'event_label': 1},
                      {'event_onset': 328, 'event_offset': 335, 'event_label': 1},
                      {'event_onset': 364, 'event_offset': 369, 'event_label': 1},
                      {'event_onset': 368, 'event_offset': 375, 'event_label': 1},
                      {'event_onset': 442, 'event_offset': 447, 'event_label': 1},
                      {'event_onset': 454, 'event_offset': 461, 'event_label': 1},
                      {'event_onset': 464, 'event_offset': 469, 'event_label': 1},
                      {'event_onset': 476, 'event_offset': 481, 'event_label': 1},
                      {'event_onset': 480, 'event_offset': 487, 'event_label': 1},
                      {'event_onset': 486, 'event_offset': 493, 'event_label': 1},
                      {'event_onset': 492, 'event_offset': 505, 'event_label': 1},
                      {'event_onset': 518, 'event_offset': 523, 'event_label': 1},
                      {'event_onset': 530, 'event_offset': 537, 'event_label': 1},
                      {'event_onset': 550, 'event_offset': 555, 'event_label': 1},
                      {'event_onset': 576, 'event_offset': 591, 'event_label': 1},
                      {'event_onset': 592, 'event_offset': 597, 'event_label': 1}]

    eval_result_path = 'data/COAS_2/Eval_test'

    evaluation = Evaluation(reverse_label_dict, 1.0, reference_event_list_all, estimated_event_list_all, eval_result_path)
    eval_result = evaluation.get_metrics()
    evaluation.plot_metrics(eval_result)
    confusion_matrix, event_labels= evaluation.get_confusion_matrix(evaluated_length_seconds=None)
    evaluation.plot_confusion_matrix(confusion_matrix, event_labels)