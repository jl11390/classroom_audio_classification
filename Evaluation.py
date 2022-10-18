import json
import sed_eval

class Evaluation:
    def __init__(self, estimated_event_list, test_audio, annot_path):
        self.estimated_event_list = estimated_event_list
        self.test_audio = test_audio
        self.annot_path = annot_path
    
    def evaluate_single_file(self):
        """evaluate the estimated_event_list against reference_event_list using sed_eval"""
        with open(self.annot_path, 'r') as f:
            annot = json.load(f)
        test_name = self.test_audio.split('/')[-1]
        for file in annot:
            file_name_mp4 = file['video_url'].split('-')[-1]
            file_name = file_name_mp4.replace('.mp4', '.wav')
            if file_name == test_name:
                reference_event_list = file['tricks']
        assert reference_event_list is not None, 'test audio is not in annotation file'
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
        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list,time_resolution=1.0) # adjust parameter
        segment_based_metrics.evaluate(reference_event_list,self.estimated_event_list)
        # Get all metrices
        all_class_wise_metrics = segment_based_metrics.results_class_wise_metrics()
        # Filter metrices and change output format
        result = {}
        for c in list(all_class_wise_metrics.items()):
            result[c[0]] = {'0':0}
            result[c[0]]['f_measure'] = c[1]['f_measure']['f_measure']
            result[c[0]]['precision'] = c[1]['f_measure']['precision']
            result[c[0]]['recall'] = c[1]['f_measure']['recall']
            result[c[0]]['error_rate'] = c[1]['error_rate']['error_rate']
            del result[c[0]]['0']
        return result

if __name__ == '__main__':
    estimated_event_list = [{'event_onset': 0, 'event_offset': 74, 'event_label': 'Lecturing'}, {'event_onset': 100, 'event_offset': 140, 'event_label': 'Lecturing'}, {'event_onset': 176, 'event_offset': 246, 'event_label': 'Lecturing'}, {'event_onset': 328, 'event_offset': 430, 'event_label': 'Lecturing'}]
    annot_path = 'data/COAS/Annotation/project-3-at-2022-10-10-17-01-baad4ee5.json'
    test_audio = 'data/COAS/TestAudios/Technology_1_008.wav'
    evaluation = Evaluation(estimated_event_list, test_audio, annot_path)
    result = evaluation.evaluate_single_file()
    print(result)