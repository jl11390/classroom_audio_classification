import sed_eval
import public_func as f
import numpy as np
from collections import defaultdict


class Evaluation:
    def __init__(self, time_resolution=1.0, target_class_version=0):
        self.time_resolution = time_resolution
        self.target_class_version = target_class_version
        self.label_dict = f.get_label_dict(target_class_version)
        self.reverse_label_dict = f.get_reverse_label_dict(target_class_version)

    def evaluate_single_file(self, estimated_event_list, reference_event_list):
        """evaluate the estimated_event_list against reference_event_list using sed_eval"""
        for event_dict in reference_event_list:
            event_dict['event_onset'] = event_dict['start']
            event_dict['event_offset'] = event_dict['end']
            event_dict['event_label'] = self.label_dict[event_dict['labels'][0]]
            del event_dict['start']
            del event_dict['end']
            del event_dict['labels']
        for event_dict in estimated_event_list:
            event_dict['event_label'] = self.label_dict[event_dict['event_label']]

        event_label_list = list(set(val for val in self.label_dict.values())
)
        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list,
                                                                         time_resolution=self.time_resolution)  # adjust parameter
        segment_based_metrics.evaluate(reference_event_list, estimated_event_list)

        # Get all metrices
        all_class_wise_metrics = segment_based_metrics.results_class_wise_metrics()
        # Filter metrices and change output format
        eval_result = {}
        for label_class, result in list(all_class_wise_metrics.items()):
            label = self.reverse_label_dict[label_class]
            eval_result[label] = {999: '999'}
            eval_result[label]['f_measure'] = result['f_measure']['f_measure'] if result['f_measure']['f_measure'] > 0 else 0.0
            eval_result[label]['precision'] = result['f_measure']['precision'] if result['f_measure']['precision'] > 0 else 0.0
            eval_result[label]['recall'] = result['f_measure']['recall'] if result['f_measure']['recall'] > 0 else 0.0
            eval_result[label]['error_rate'] = result['error_rate']['error_rate'] if result['error_rate']['error_rate'] > 0 else 0.0
            del eval_result[label][999]

        return eval_result


if __name__ == '__main__':
    estimated_event_list = [{'event_onset': 0, 'event_offset': 74, 'event_label': 'Lecturing'},
                            {'event_onset': 100, 'event_offset': 140, 'event_label': 'Lecturing'},
                            {'event_onset': 176, 'event_offset': 246, 'event_label': 'Lecturing'},
                            {'event_onset': 328, 'event_offset': 430, 'event_label': 'Lecturing'}]
    reference_event_list = [
        {
            "start": 71.5442350718065,
            "end": 90.58838397581255,
            "labels": [
                "Collaborative Student Work"
            ]
        },
        {
            "start": 89.55897052154195,
            "end": 267.64749811035523,
            "labels": [
                "Lecturing"
            ]
        },
        {
            "start": 267.13279138321997,
            "end": 346.39762736205597,
            "labels": [
                "Q/A"
            ]
        },
        {
            "start": 344.85350718065007,
            "end": 451.3977996976568,
            "labels": [
                "Lecturing"
            ]
        }
    ]
    evaluation = Evaluation()
    result = evaluation.evaluate_single_file(estimated_event_list, reference_event_list)
    print(result)
