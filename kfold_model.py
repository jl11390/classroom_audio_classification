from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import RandomForestClassifier
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
    print(fold_acc)
