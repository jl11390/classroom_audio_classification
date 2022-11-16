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

    # Version 0: without data augmentation and multi-scaling, predict threshold
    cased0 = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased0.load_train_data(annot_path, audio_path, cache_path, audio_aug_path=False, cache_aug_path=False,
                          aug_dict_path=False, load_cache=True, num_folds=5, multi_scaling=False)
    print(f"V0 \nfeatures matrix shape: {cased0.features_matrix_all.shape} \n labels matrix shape: {cased0.labels_matrix_all.shape}")

    cased0.randomized_search_cv(n_iter_search=10, cache_path=model_cache_path, load_cache=True)

    cased0.evaluate_all(annot_path, audio_test_path, cache_test_path, eval_test_path, plot=True, predict_type = 'threshold', load_cache=True)

    # Version 1: without multi-scaling, predict threshold
    cased1 = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased1.load_train_data(annot_path, audio_path, cache_path, audio_aug_path=False, cache_aug_path=False,
                          aug_dict_path=False, load_cache=True, num_folds=5, multi_scaling=True)
    print(f"V123 \nfeatures matrix shape: {cased1.features_matrix_all.shape} \n labels matrix shape: {cased1.labels_matrix_all.shape}")

    cased1.randomized_search_cv(n_iter_search=10, cache_path=model_cache_path, load_cache=True)

    cased1.evaluate_all(annot_path, audio_test_path, cache_test_path, eval_test_path, plot=True, predict_type = 'threshold', load_cache=True)

    # Version 2: without multi-scaling, predict viterbi without pstate
    cased2 = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased2.load_train_data(annot_path, audio_path, cache_path, audio_aug_path=False, cache_aug_path=False,
                          aug_dict_path=False, load_cache=True, num_folds=5, multi_scaling=True)
    print(f"V123 \nfeatures matrix shape: {cased2.features_matrix_all.shape} \n labels matrix shape: {cased2.labels_matrix_all.shape}")

    cased2.randomized_search_cv(n_iter_search=10, cache_path=model_cache_path, load_cache=True)

    cased2.evaluate_all(annot_path, audio_test_path, cache_test_path, eval_test_path, plot=True, predict_type = 'viterbi_without_pstate', load_cache=True)

    # Version 3: without multi-scaling, predict viterbi with pstate
    cased3 = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased3.load_train_data(annot_path, audio_path, cache_path, audio_aug_path=False, cache_aug_path=False,
                          aug_dict_path=False, load_cache=True, num_folds=5, multi_scaling=True)
    print(f"V123 \nfeatures matrix shape: {cased3.features_matrix_all.shape} \n labels matrix shape: {cased3.labels_matrix_all.shape}")

    cased3.randomized_search_cv(n_iter_search=10, cache_path=model_cache_path, load_cache=True)

    cased3.evaluate_all(annot_path, audio_test_path, cache_test_path, eval_test_path, plot=True, predict_type = 'viterbi_with_pstate', load_cache=True)


    # Version 4: with data augmentation and multi-scaling
    cased4 = CASED(frac_t, long_frac_t, long_long_frac_t, step_t, target_class_version=0)

    cased4.load_train_data(annot_path, audio_path, cache_path, audio_aug_path=audio_aug_path, cache_aug_path=cache_aug_path,
                          aug_dict_path=aug_dict_path, load_cache=True, num_folds=5, multi_scaling=True)
    print(f"V4 \nfeatures matrix shape: {cased4.features_matrix_all.shape} \n labels matrix shape: {cased4.labels_matrix_all.shape}")

    cased4.randomized_search_cv(n_iter_search=10, cache_path=model_cache_path, load_cache=True)

    cased4.evaluate_all(annot_path, audio_test_path, cache_test_path, eval_test_path, plot=True, predict_type = 'viterbi_with_pstate', load_cache=True)
