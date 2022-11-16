from CASED import CASED
import warnings

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

    # Version 0: without data augmentation and multi-scaling
    cased = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased.load_train_data(annot_path, audio_path, cache_path, audio_aug_path=False, cache_aug_path=False,
                          aug_dict_path=False, load_cache=True, num_folds=5, multi_scaling=False)
    print(f"V0 \nfeatures matrix shape: {cased.features_matrix_all.shape} \n labels matrix shape: {cased.labels_matrix_all.shape}")

    # Version 1,2,3: without data augmentation
    cased = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased.load_train_data(annot_path, audio_path, cache_path, audio_aug_path=False, cache_aug_path=False,
                          aug_dict_path=False, load_cache=True, num_folds=5, multi_scaling=True)
    print(f"V123 \nfeatures matrix shape: {cased.features_matrix_all.shape} \n labels matrix shape: {cased.labels_matrix_all.shape}")

    # Version 4: with data augmentation and multi-scaling
    cased = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased.load_train_data(annot_path, audio_path, cache_path, audio_aug_path=audio_aug_path, cache_aug_path=cache_aug_path,
                          aug_dict_path=aug_dict_path, load_cache=True, num_folds=5, multi_scaling=True)
    print(f"V4 \nfeatures matrix shape: {cased.features_matrix_all.shape} \n labels matrix shape: {cased.labels_matrix_all.shape}")