from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import RandomForestClassifier
from librosa.sequence import viterbi_binary
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class KfoldModel:
    def __init__(self, feature_matrix, labels, folds, cfg):
        self.X = feature_matrix
        self.y = labels
        self.folds = folds
        self.cfg = cfg
        self.val_fold_scores_ = []

    def train_kfold(self):
        logo = LeaveOneGroupOut()
        for train_index, test_index in logo.split(self.X, self.y, self.folds):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            ss = StandardScaler(copy=True)
            X_train = ss.fit_transform(X_train)
            X_test = ss.transform(X_test)

            clf = self.cfg["model"]
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            # In multilabel classification, this function computes subset accuracy:
            # the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
            fold_acc = accuracy_score(y_test, y_pred)
            self.val_fold_scores_.append(fold_acc)

            y_pred_prob = clf.predict_proba(X_test)
            prob = np.array([y_pred_prob_label[:,1] for y_pred_prob_label in y_pred_prob])
            a = np.array([[0.95, 0.05], [0.05, 0.95]])
            transition = np.repeat(a[np.newaxis, :, :], 4, axis=0)
            binary_pred = viterbi_binary(prob, transition)
            # TODO: write a def or class to evaluate the performance of binary prediction

        return self.val_fold_scores_

if __name__ == "__main__":
    cache_path = 'data/COAS/Features/preprocessing.pkl'
    with open(cache_path, 'rb') as f:
        features_labels_folds = pickle.load(f)
    feature_matrix, labels_matrix, folds = features_labels_folds
    model_cfg = dict(
        model=RandomForestClassifier(
            random_state=42,
            n_jobs=10,
            class_weight="balanced",
            n_estimators=500,
            bootstrap=True,
        ),
    )
    model = KfoldModel(feature_matrix, labels_matrix, folds, model_cfg)
    fold_acc = model.train_kfold()