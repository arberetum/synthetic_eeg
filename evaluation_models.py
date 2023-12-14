from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

from feature_selection import rank_correlation


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
    print(gcv.best_params_)

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


def evaluate_synthetic_naive_bayes(train_x_real, train_y_real, train_x_fake, train_y_fake, test_x, test_y):
    synthetic_ratios = np.arange(0, 1.01, 0.1)
    macro_f1s = []
    binary_f1s = []
    aucs = []
    for ratio in synthetic_ratios:
        if ratio == 0:
            train_x = train_x_real
            train_y = train_y_real
        elif ratio == 1:
            train_x = train_x_fake
            train_y = train_y_fake
        else:
            train_x_real_sample, _, train_y_real_sample, _ = train_test_split(train_x_real, train_y_real,
                                                                              train_size=1.0-ratio, stratify=train_y_real)
            train_x_fake_sample, _, train_y_fake_sample, _ = train_test_split(train_x_fake, train_y_fake,
                                                                              train_size=ratio, stratify=train_y_fake)
            train_x = np.concatenate((train_x_real_sample, train_x_fake_sample))
            train_y = np.concatenate((train_y_real_sample, train_y_fake_sample))

        results = naive_bayes(train_x, train_y, test_x, test_y)
        macro_f1s.append(results['test-macro-f1'])
        binary_f1s.append(results['test-f1'])
        aucs.append(results['test-auc'])

    fig = plt.figure(figsize=(4, 4))
    plt.plot(synthetic_ratios, macro_f1s, label='Macro F1')
    plt.plot(synthetic_ratios, binary_f1s, label='Binary F1')
    plt.plot(synthetic_ratios, aucs, label='AUC')
    plt.legend(fontsize='large')
    plt.xlabel("Ratio of Synthetic Data", fontsize='large')
    plt.savefig("./outputs/DDPM/NaiveBayesMetrics_ddpm.png")
    plt.show()

    ratio_results = {'macro-f1s': macro_f1s, 'f1s': binary_f1s, 'aucs': aucs, 'ratios': synthetic_ratios}
    return ratio_results


def evaluate_synthetic_adaboost(train_x_real, train_y_real, train_x_fake, train_y_fake, test_x, test_y):
    synthetic_ratios = np.arange(0, 1.01, 0.1)
    macro_f1s = []
    binary_f1s = []
    aucs = []
    for ratio in synthetic_ratios:
        if ratio == 0:
            train_x = train_x_real
            train_y = train_y_real
        elif ratio == 1:
            train_x = train_x_fake
            train_y = train_y_fake
        else:
            train_x_real_sample, _, train_y_real_sample, _ = train_test_split(train_x_real, train_y_real,
                                                                              train_size=1.0-ratio, stratify=train_y_real)
            train_x_fake_sample, _, train_y_fake_sample, _ = train_test_split(train_x_fake, train_y_fake,
                                                                              train_size=ratio, stratify=train_y_fake)
            train_x = np.concatenate((train_x_real_sample, train_x_fake_sample))
            train_y = np.concatenate((train_y_real_sample, train_y_fake_sample))

        ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)

        ada.fit(train_x, train_y)
        results = get_metrics(ada, train_x, train_y, test_x, test_y)

        macro_f1s.append(results['test-macro-f1'])
        binary_f1s.append(results['test-f1'])
        aucs.append(results['test-auc'])

    fig = plt.figure(figsize=(4, 4))
    plt.plot(synthetic_ratios, macro_f1s, label='Macro F1')
    plt.plot(synthetic_ratios, binary_f1s, label='Binary F1')
    plt.plot(synthetic_ratios, aucs, label='AUC')
    plt.legend(fontsize='large')
    plt.xlabel("Ratio of Synthetic Data", fontsize='large')
    plt.savefig("./outputs/DDPM/AdaBoostMetrics_ddpm.png")
    plt.show()

    ratio_results = {'macro-f1s': macro_f1s, 'f1s': binary_f1s, 'aucs': aucs, 'ratios': synthetic_ratios}
    return ratio_results


if __name__ == '__main__':
    # test_feat_folder = "/Volumes/LACIE SHARE/seizure-prediction-outputs/test_windows_3sduration_1soverlap_features"

    # # split into train and test
    # test_files = [f.strip() for f in open("./data/test.txt").readlines() if "Dog_5" not in f]
    # test_feat_files = [f.split("/")[-1].split(".")[0] + "_features.csv" for f in test_files]
    # test_classes = np.zeros((len(test_feat_files, )))
    # for i, filename in enumerate(test_feat_files):
    #     if "preictal" in filename:
    #         test_classes[i] = 1
    #
    # X_train, X_test = train_test_split(test_feat_files, test_size=1.0/3, random_state=8, stratify=test_classes)

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
    # train_x = np.load("train_x_3s.npy")
    # train_y = np.load("train_y_3s.npy")
    # test_x = np.load("test_x_3s.npy")
    # test_y = np.load("test_y_3s.npy")
    # print(type(train_y))
    # print(type(test_y))

    # print(naive_bayes(train_x, train_y, test_x, test_y))
    # print(ridge_classifier(train_x, train_y, test_x, test_y))
    # print(adaboost(train_x, train_y, test_x, test_y))

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
    # train_x = np.load("train_x_10m.npy")
    # train_y = np.load("train_y_10m.npy")
    # test_x = np.load("test_x_10m.npy")
    # test_y = np.load("test_y_10m.npy")
    #
    # print(naive_bayes(train_x, train_y, test_x, test_y))
    # # {'train-macro-f1': 0.3022503902923645, 'train-f1': 0.15498154981549817, 'train-precision': 0.08484848484848485, 'train-recall': 0.8936170212765957, 'test-macro-f1': 0.2663817285654871, 'test-f1': 0.11070110701107011, 'test-precision': 0.06048387096774194, 'test-recall': 0.6521739130434783, 'train-auc': 0.5914893617021276, 'test-auc': 0.4582825409725044}
    # print(ridge_classifier(train_x, train_y, test_x, test_y))
    # print(adaboost(train_x, train_y, test_x, test_y))

    # combine real training features
    # eval_train_files = [f.strip() for f in open("./data/eval_train_files.txt").readlines()]
    # eval_test_files = [f.strip() for f in open("./data/eval_test_files.txt").readlines()]
    # feature_folder = "/Volumes/LACIE SHARE/seizure-prediction-outputs/real_10s_segments_windows_1sdur_1soverlap"
    # for i, filename in enumerate(eval_train_files):
    #     feature_file_name = filename.split("/")[-1]
    #     feature_file_name = feature_file_name.split(".")[0] + "_features.csv"
    #     feature_file_path = os.path.join(feature_folder, feature_file_name)
    #
    #     if i == 0:
    #         eval_train_df = pd.read_csv(feature_file_path)
    #     else:
    #         this_df = pd.read_csv(feature_file_path)
    #         eval_train_df = pd.concat((eval_train_df, this_df))
    # print(eval_train_df)
    # eval_train_df_y = eval_train_df['class']
    # eval_train_df.drop(columns=['class'], inplace=True)
    # with open("eval_real_train_x.npy", 'wb') as f:
    #     np.save(f, eval_train_df.to_numpy())
    # with open("eval_real_train_y.npy", 'wb') as f:
    #     np.save(f, eval_train_df_y.to_numpy())

    # for i, filename in enumerate(eval_test_files):
    #     feature_file_name = filename.split("/")[-1]
    #     feature_file_name = feature_file_name.split(".")[0] + "_features.csv"
    #     feature_file_path = os.path.join(feature_folder, feature_file_name)
    #
    #     if i == 0:
    #         eval_test_df = pd.read_csv(feature_file_path)
    #     else:
    #         this_df = pd.read_csv(feature_file_path)
    #         eval_test_df = pd.concat((eval_test_df, this_df))
    # print(eval_test_df)
    # eval_test_df_y = eval_test_df['class']
    # eval_test_df.drop(columns=['class'], inplace=True)
    # with open("eval_real_test_x.npy", 'wb') as f:
    #     np.save(f, eval_test_df.to_numpy())
    # with open("eval_real_test_y.npy", 'wb') as f:
    #     np.save(f, eval_test_df_y.to_numpy())


    # eval_train_x = np.load("eval_real_train_x.npy")
    # eval_train_y = np.load("eval_real_train_y.npy")
    # eval_test_x = np.load("eval_real_test_x.npy")
    # eval_test_y = np.load("eval_real_test_y.npy")

    # percent_top_feat_to_keep = 0.8
    # feat_to_keep = int(percent_top_feat_to_keep * eval_train_x.shape[1])
    # print(f"{feat_to_keep} features kept")
    # sorted_inds, rank_values = rank_correlation(eval_train_x, eval_train_y)
    # eval_train_x_selected = eval_train_x[:, sorted_inds[:feat_to_keep]]
    # eval_test_x_selected = eval_test_x[:, sorted_inds[:feat_to_keep]]
    # print(f"Shape of train x: {eval_train_x_selected.shape}")
    # print(f"Shape of test x: {eval_test_x_selected.shape}")

    # print(naive_bayes(eval_train_x_selected, eval_train_y, eval_test_x_selected, eval_test_y))
    # print(ridge_classifier(eval_train_x_selected, eval_train_y, eval_test_x_selected, eval_test_y))
    # print(adaboost(eval_train_x_selected, eval_train_y, eval_test_x_selected, eval_test_y))

    # num_preictal_test = np.sum(eval_test_y)
    # print(f"Number of preictal test samples: {num_preictal_test}")
    # print(f"Number of interictal test samples: {eval_test_y.shape[0] - num_preictal_test}")


    # # WGAN evaluation
    # eval_train_x = np.load("eval_real_train_x.npy")
    # eval_train_y = np.load("eval_real_train_y.npy")
    # eval_test_x = np.load("eval_real_test_x.npy")
    # eval_test_y = np.load("eval_real_test_y.npy")
    # wgan_train_x_df = pd.read_csv(
    #     "/Volumes/LACIE SHARE/seizure-prediction-outputs/wgan_10s_segments_windows_1sdur_1soverlap/interictal_set1_features.csv")
    # wgan_train_x_df = pd.concat((wgan_train_x_df, pd.read_csv(
    #     "/Volumes/LACIE SHARE/seizure-prediction-outputs/wgan_10s_segments_windows_1sdur_1soverlap/interictal_set2_features.csv")))
    # wgan_train_x_df = pd.concat((wgan_train_x_df, pd.read_csv(
    #     "/Volumes/LACIE SHARE/seizure-prediction-outputs/wgan_10s_segments_windows_1sdur_1soverlap/interictal_set3_features.csv")))
    # wgan_train_x_df = pd.concat((wgan_train_x_df, pd.read_csv(
    #     "/Volumes/LACIE SHARE/seizure-prediction-outputs/wgan_10s_segments_windows_1sdur_1soverlap/preictal_set1_features.csv")))
    # wgan_train_x_df = pd.concat((wgan_train_x_df, pd.read_csv(
    #     "/Volumes/LACIE SHARE/seizure-prediction-outputs/wgan_10s_segments_windows_1sdur_1soverlap/preictal_set2_features.csv")))
    # wgan_train_x_df = pd.concat((wgan_train_x_df, pd.read_csv(
    #     "/Volumes/LACIE SHARE/seizure-prediction-outputs/wgan_10s_segments_windows_1sdur_1soverlap/preictal_set3_features.csv")))
    # wgan_train_y = wgan_train_x_df['class'].to_numpy()
    # wgan_train_x = wgan_train_x_df.drop(columns=['class']).to_numpy()
    #
    # percent_top_feat_to_keep = 0.8
    # feat_to_keep = int(percent_top_feat_to_keep * eval_train_x.shape[1])
    # print(f"{feat_to_keep} features kept")
    # sorted_inds, rank_values = rank_correlation(eval_train_x, eval_train_y)
    # eval_train_x_selected = eval_train_x[:, sorted_inds[:feat_to_keep]]
    # wgan_train_x_selected = wgan_train_x[:, sorted_inds[:feat_to_keep]]
    # eval_test_x_selected = eval_test_x[:, sorted_inds[:feat_to_keep]]
    # print(f"Shape of train x: {eval_train_x_selected.shape}")
    # print(f"Shape of wgan x: {wgan_train_x_selected.shape}")
    # print(f"Shape of test x: {eval_test_x_selected.shape}")
    #
    # # ratio_results = evaluate_synthetic_naive_bayes(eval_train_x_selected, eval_train_y,
    # #                                                wgan_train_x_selected, wgan_train_y,
    # #                                                eval_test_x_selected, eval_test_y)
    # #
    # # with open("./outputs/WGAN/naive_bayes_eval.pkl", 'wb') as f:
    # #     pickle.dump(ratio_results, f)
    #
    # ratio_results = evaluate_synthetic_adaboost(eval_train_x_selected, eval_train_y,
    #                                                wgan_train_x_selected, wgan_train_y,
    #                                                eval_test_x_selected, eval_test_y)
    #
    # with open("./outputs/WGAN/adaboost_eval.pkl", 'wb') as f:
    #     pickle.dump(ratio_results, f)

    # DDPM evaluation
    eval_train_x = np.load("eval_real_train_x.npy")
    eval_train_y = np.load("eval_real_train_y.npy")
    eval_test_x = np.load("eval_real_test_x.npy")
    eval_test_y = np.load("eval_real_test_y.npy")
    ddpm_train_x_df = pd.read_csv(
        "/Volumes/LACIE SHARE/seizure-prediction-outputs/ddpm_10s_segments_windows_1sdur_1soverlap/generated_samples_interictal_ddpm_features.csv")
    ddpm_train_x_df = pd.concat((ddpm_train_x_df, pd.read_csv(
        "/Volumes/LACIE SHARE/seizure-prediction-outputs/ddpm_10s_segments_windows_1sdur_1soverlap/generated_samples_preictal_ddpm_features.csv")))
    ddpm_train_y = ddpm_train_x_df['class'].to_numpy()
    ddpm_train_x = ddpm_train_x_df.drop(columns=['class']).to_numpy()

    percent_top_feat_to_keep = 0.8
    feat_to_keep = int(percent_top_feat_to_keep * eval_train_x.shape[1])
    print(f"{feat_to_keep} features kept")
    sorted_inds, rank_values = rank_correlation(eval_train_x, eval_train_y)
    eval_train_x_selected = eval_train_x[:, sorted_inds[:feat_to_keep]]
    ddpm_train_x_selected = ddpm_train_x[:, sorted_inds[:feat_to_keep]]
    eval_test_x_selected = eval_test_x[:, sorted_inds[:feat_to_keep]]
    print(f"Shape of train x: {eval_train_x_selected.shape}")
    print(f"Shape of ddpm x: {ddpm_train_x_selected.shape}")
    print(f"Shape of test x: {eval_test_x_selected.shape}")

    # ratio_results = evaluate_synthetic_naive_bayes(eval_train_x_selected, eval_train_y,
    #                                                ddpm_train_x_selected, ddpm_train_y,
    #                                                eval_test_x_selected, eval_test_y)
    #
    # with open("./outputs/DDPM/naive_bayes_eval.pkl", 'wb') as f:
    #     pickle.dump(ratio_results, f)
    #
    # ratio_results = evaluate_synthetic_adaboost(eval_train_x_selected, eval_train_y,
    #                                             ddpm_train_x_selected, ddpm_train_y,
    #                                             eval_test_x_selected, eval_test_y)
    #
    # with open("./outputs/DDPM/adaboost_eval.pkl", 'wb') as f:
    #     pickle.dump(ratio_results, f)

    print(f"DDPM All Synthetic Naive")
    nb = GaussianNB()
    nb.fit(ddpm_train_x_selected, ddpm_train_y)
    print(get_metrics(nb, ddpm_train_x_selected, ddpm_train_y, eval_test_x_selected, eval_test_y))

    print(f"DDPM + Real Naive")
    nb = GaussianNB()
    train_x = np.concatenate((ddpm_train_x_selected, eval_test_x_selected))
    train_y = np.concatenate((ddpm_train_y, eval_test_y))
    nb.fit(train_x, train_y)
    print(get_metrics(nb, train_x, train_y, eval_test_x_selected, eval_test_y))

    print(f"DDPM All Synthetic Ada")
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    ada.fit(ddpm_train_x_selected, ddpm_train_y)
    print(get_metrics(ada, ddpm_train_x_selected, ddpm_train_y, eval_test_x_selected, eval_test_y))

    print(f"DDPM + Real Ada")
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    train_x = np.concatenate((ddpm_train_x_selected, eval_test_x_selected))
    train_y = np.concatenate((ddpm_train_y, eval_test_y))
    ada.fit(train_x, train_y)
    print(get_metrics(ada, train_x, train_y, eval_test_x_selected, eval_test_y))