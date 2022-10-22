import sed_eval


class Evaluation:
    def __init__(self, time_resolution=1.0):
        self.time_resolution = time_resolution

    def evaluate_single_file(self, estimated_event_list, reference_event_list):
        """evaluate the estimated_event_list against reference_event_list using sed_eval"""
        event_label_list = []
        for dict in reference_event_list:
            dict['event_onset'] = dict['start']
            dict['event_offset'] = dict['end']
            dict['event_label'] = dict['labels'][0]
            event_label_list.append(dict['event_label'])
            del dict['start']
            del dict['end']
            del dict['labels']
        event_label_list = list(set(event_label_list))
        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list,
                                                                         time_resolution=self.time_resolution)  # adjust parameter
        segment_based_metrics.evaluate(reference_event_list, estimated_event_list)
        # Get all metrices
        all_class_wise_metrics = segment_based_metrics.results_class_wise_metrics()
        # Filter metrices and change output format
        result = {}
        for c in list(all_class_wise_metrics.items()):
            result[c[0]] = {'0': 0}
            result[c[0]]['f_measure'] = c[1]['f_measure']['f_measure']
            result[c[0]]['precision'] = c[1]['f_measure']['precision']
            result[c[0]]['recall'] = c[1]['f_measure']['recall']
            result[c[0]]['error_rate'] = c[1]['error_rate']['error_rate']
            del result[c[0]]['0']
        return result


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
