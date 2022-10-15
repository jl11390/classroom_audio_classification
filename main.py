from data_loader import DataLoader
from kfold_model import KfoldModel
from sklearn.ensemble import RandomForestClassifier
from model import Model
import numpy as np
import json
import warnings


############################
# use k-fold to select best model params on the train_val data
############################
def select_model(annot_path, audio_path, cache_path, frac_t, step_t, load_cache=False, num_folds=5, train_val_per=0.9):
    with open(annot_path, 'r') as f:
        annot = json.load(f)
    f.close()

    n = len(annot)
    n_train_val = int(n*train_val_per)
    print(f"{n_train_val} audios feed into the kfold pipeline")
    features_matrix_all = None
    labels_matrix_all = None
    folds_all = []
    for i, metadict in enumerate(annot):
        if i < n_train_val:
            file_name_mp4 = metadict['video_url'].split('-')[-1]
            file_name = file_name_mp4.replace('.mp4', '.wav')
            fold = i % num_folds
            dataloader = DataLoader(file_name, audio_path, cache_path, metadict, frac_t, step_t)
            features_matrix, labels_matrix = dataloader.load_data(load_cache = load_cache)
            folds = labels_matrix.shape[0]*[fold]
            features_matrix_all = np.vstack([features_matrix_all, features_matrix]) if features_matrix_all is not None else features_matrix
            labels_matrix_all = np.vstack([labels_matrix_all, labels_matrix]) if labels_matrix_all is not None else labels_matrix
            folds_all.extend(folds)
            print(f"loaded {i+1} audios and {n_train_val-i-1} to go")
    folds_all = np.array(folds_all)
    print(f'proportion of labels: {np.sum(labels_matrix_all, axis=0)/np.sum(labels_matrix_all)}')

    model_cfg = dict(
        model=RandomForestClassifier(
            random_state=42,
            n_jobs=10,
            class_weight="balanced",
            n_estimators=500,
            bootstrap=True,
        ),
    )
    ################################
    # TODO: model selection class here
    ################################
    kfoldmodel = KfoldModel(features_matrix_all, labels_matrix_all, folds_all, model_cfg)
    fold_acc = kfoldmodel.train_kfold()
    print(f'best kfold performance: {fold_acc}')

    return model_cfg

############################
# train model with all train_val data and evaluate performance on hold out samples
############################
def train_and_evaluate(model_cfg, annot_path, audio_path, cache_path, frac_t, step_t, load_cache=True, train_val_per=0.9):
    with open(annot_path, 'r') as f:
        annot = json.load(f)
    f.close()

    n = len(annot)
    n_train_val = int(n*train_val_per)
    print(f"{n_train_val} audios feed into the training pipeline")
    features_matrix_all = None
    labels_matrix_all = None
    for i, metadict in enumerate(annot):
        if i < n_train_val:
            file_name_mp4 = metadict['video_url'].split('-')[-1]
            file_name = file_name_mp4.replace('.mp4', '.wav')
            dataloader = DataLoader(file_name, audio_path, cache_path, metadict, frac_t, step_t)
            features_matrix, labels_matrix = dataloader.load_data(load_cache = load_cache)
            features_matrix_all = np.vstack([features_matrix_all, features_matrix]) if features_matrix_all is not None else features_matrix
            labels_matrix_all = np.vstack([labels_matrix_all, labels_matrix]) if labels_matrix_all is not None else labels_matrix
            print(f"loaded {i+1} audios and {n_train_val-i-1} to go")
    print(f'proportion of labels: {np.sum(labels_matrix_all, axis=0)/np.sum(labels_matrix_all)}')

    model = Model(model_cfg)
    model.train(features_matrix_all, labels_matrix_all)
    evaluation_results_all = []
    for i, metadict in enumerate(annot):
        if i >= n_train_val:
            file_name_mp4 = metadict['video_url'].split('-')[-1]
            file_name = file_name_mp4.replace('.mp4', '.wav')
            dataloader = DataLoader(file_name, audio_path, cache_path, metadict, frac_t, step_t)
            features_matrix, labels_matrix = dataloader.load_data(load_cache = load_cache)
            evaluation_results = model.evaluate(features_matrix, metadict)
            evaluation_results_all.append(evaluation_results)

    return evaluation_results_all

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    frac_t, step_t = 10, 2
    annot_path = 'data/COAS/Annotation/project-3-at-2022-10-10-17-01-baad4ee5.json'
    audio_path = 'data/COAS/Audios'
    cache_path = 'data/COAS/Features'
    train_val_per = 0.9
    num_folds = 5

    best_model = select_model(annot_path, audio_path, cache_path, frac_t, step_t, load_cache=True, num_folds=num_folds, train_val_per=train_val_per)
    evaluation_results_all = train_and_evaluate(best_model, annot_path, audio_path, cache_path, frac_t, step_t, load_cache=True, train_val_per=train_val_per)
    for evaluation_results in evaluation_results_all:
        print(evaluation_results.shape)