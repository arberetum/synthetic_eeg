from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
import os


macro_f1_scorer = make_scorer(f1_score, average='macro')


def get_metrics(model, train_x, train_y, test_x, test_y, auc=True):
    train_preds = model.predict(train_x)
    test_preds = model.predict(test_x)

    results = dict()

    results['train-macro-f1'] = f1_score(train_y, train_preds, average='macro')
    results['train-f1'] = f1_score(train_y, train_preds)
    results['train-precision'] = precision_score(train_y, train_preds)
    results['train-recall'] = recall_score(train_y, train_preds)


    results['test-macro-f1'] = f1_score(test_y, test_preds, average='macro')
    results['test-f1'] = f1_score(test_y, test_preds)
    results['test-precision'] = precision_score(test_y, test_preds)
    results['test-recall'] = recall_score(test_y, test_preds)

    if auc:
        train_probs = model.predict_proba(train_x)
        test_probs = model.predict_proba(test_x)
        results['train-auc'] = roc_auc_score(train_y, train_probs[:, 1])
        results['test-auc'] = roc_auc_score(test_y, test_probs[:, 1])

    return results


def naive_bayes(train_x, train_y, test_x, test_y):
    model = GaussianNB()
    model.fit(train_x, train_y)
    return get_metrics(model, train_x, train_y, test_x, test_y)


def ridge_classifier(train_x, train_y, test_x, test_y):
    alphas = [10.0**x for x in range(-5, 2)]

    rcv = RidgeClassifierCV(alphas=alphas, cv=5, scoring=macro_f1_scorer)
    rcv.fit(train_x, train_y)

    return get_metrics(rcv, train_x, train_y, test_x, test_y, auc=False)


def adaboost(train_x, train_y, test_x, test_y):
    n_estimators = [5, 20, 50, 100]
    learning_rates = [10.0**x for x in range(-5, 2)]

    ada = AdaBoostClassifier()

    gcv = GridSearchCV(ada, param_grid={'n_estimators': n_estimators, 'learning_rate': learning_rates},
                       scoring=macro_f1_scorer, n_jobs=-2, cv=5)

    fit_ada = gcv.fit(train_x, train_y)

    return get_metrics(fit_ada, train_x, train_y, test_x, test_y)


def load_data(folder, filenames):
    y = np.array([])
    for i, filename in enumerate(filenames):
        features = pd.read_csv(os.path.join(folder, filename)).to_numpy()
        if i == 0:
            x = features
            # print(filename)
        else:
            try:
                x = np.concatenate((x, features), axis=0)
            except:
                print(x.shape)
                print(features.shape)
                print(filename)
                raise
                continue

        if "preictal" in filename:
            y = np.concatenate((y, np.ones(features.shape[0],)))
        elif "interictal" in filename:
            y = np.concatenate((y, np.zeros(features.shape[0],)))
        else:
            raise

    return x, y


def load_and_flatten_data(folder, filenames):
    y = np.zeros((len(filenames),))
    for i, filename in enumerate(filenames):
        features = pd.read_csv(os.path.join(folder, filename)).to_numpy()
        features = features.flatten().reshape(1, -1)
        if i == 0:
            x = features
        else:
            try:
                x = np.concatenate((x, features), axis=0)
            except:
                print(x.shape)
                raise

        if "preictal" in filename:
            y[i] = 1

    return x, y


if __name__ == '__main__':
    test_feat_folder = "/Volumes/LACIE SHARE/seizure-prediction-outputs/test_windows_3sduration_1soverlap_features"

    # split into train and test
    test_files = [f.strip() for f in open("./data/test.txt").readlines() if "Dog_5" not in f]
    test_feat_files = [f.split("/")[-1].split(".")[0] + "_features.csv" for f in test_files]
    test_classes = np.zeros((len(test_feat_files, )))
    for i, filename in enumerate(test_feat_files):
        if "preictal" in filename:
            test_classes[i] = 1

    X_train, X_test = train_test_split(test_feat_files, test_size=1.0/3, random_state=8, stratify=test_classes)

    # pos_samp_train = [samp for samp in X_train if "preictal" in samp]
    # neg_samp_train = [samp for samp in X_train if "interictal" in samp]
    # print(len(pos_samp_train))
    # print(len(neg_samp_train))
    # neg_sub_samp = np.random.choice(neg_samp_train, size=len(pos_samp_train), replace=False)
    # X_train = neg_sub_samp.tolist() + pos_samp_train
    # print(len(X_train))

    # print(X_test)

    # load data, separate 3s windows
    # train_x, train_y = load_data(test_feat_folder, X_train)
    # test_x, test_y = load_data(test_feat_folder, X_test)
    # (print(train_x.shape))
    # (print(train_y.shape))
    # (print(test_x.shape))
    # (print(test_y.shape))
    # np.save("train_x_3s.npy", train_x)
    # np.save("train_y_3s.npy", train_y)
    # np.save("test_x_3s.npy", test_x)
    # np.save("test_y_3s.npy", test_y)
    train_x = np.load("train_x_3s.npy")
    train_y = np.load("train_y_3s.npy")
    test_x = np.load("test_x_3s.npy")
    test_y = np.load("test_y_3s.npy")
    # print(type(train_y))
    # print(type(test_y))

    # print(naive_bayes(train_x, train_y, test_x, test_y))
    # print(ridge_classifier(train_x, train_y, test_x, test_y))
    print(adaboost(train_x, train_y, test_x, test_y))

    # # load data, separate 10 min windows
    # train_x, train_y = load_and_flatten_data(test_feat_folder, X_train)
    # test_x, test_y = load_and_flatten_data(test_feat_folder, X_test)
    # (print(train_x.shape))
    # (print(train_y.shape))
    # (print(test_x.shape))
    # (print(test_y.shape))
    # np.save("train_x_10m.npy", train_x)
    # np.save("train_y_10m.npy", train_y)
    # np.save("test_x_10m.npy", test_x)
    # np.save("test_y_10m.npy", test_y)
    train_x = np.load("train_x_10m.npy")
    train_y = np.load("train_y_10m.npy")
    test_x = np.load("test_x_10m.npy")
    test_y = np.load("test_y_10m.npy")
    #
    # print(naive_bayes(train_x, train_y, test_x, test_y))
    # # {'train-macro-f1': 0.3022503902923645, 'train-f1': 0.15498154981549817, 'train-precision': 0.08484848484848485, 'train-recall': 0.8936170212765957, 'test-macro-f1': 0.2663817285654871, 'test-f1': 0.11070110701107011, 'test-precision': 0.06048387096774194, 'test-recall': 0.6521739130434783, 'train-auc': 0.5914893617021276, 'test-auc': 0.4582825409725044}
    # print(ridge_classifier(train_x, train_y, test_x, test_y))
    print(adaboost(train_x, train_y, test_x, test_y))
