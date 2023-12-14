import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

# import evaluation_models


train_feat_folder = "/Volumes/LACIE SHARE/seizure-prediction-outputs/windows_3sduration_1soverlap_features"

def rank_correlation(x, y):
    rank_values = [0] * x.shape[1]
    for i in range(len(rank_values)):
        stacked_mat = np.asarray(np.stack((x[:, i], y), axis=1), dtype=float)
        this_cov = np.cov(stacked_mat, rowvar=False)
        rank_values[i] = this_cov[0, 1] / np.sqrt(this_cov[0, 0]*this_cov[1, 1])
    rank_values = np.array(rank_values)
    # rank_values[np.isnan(rank_values)] = 0
    sorted_inds = np.flip(np.argsort(np.abs(rank_values)))
    return sorted_inds, rank_values


def load_features(folder=train_feat_folder, flatten=False):
    files_to_load = [filename for filename in os.listdir(folder) if "._" not in filename]
    for i, filename in enumerate(files_to_load):
        this_train_x = pd.read_csv(os.path.join(folder, filename), index_col=0)
        if flatten:
            this_train_x = this_train_x.flatten().reshape(1, -1)
        if "interictal" in filename:
            this_train_y = pd.DataFrame({'class': [0] * this_train_x.shape[0]})
        else:
            this_train_y = pd.DataFrame({'class': [1] * this_train_x.shape[0]})
        if i == 0:
            train_x = this_train_x
            train_y = this_train_y
        else:
            train_x = pd.concat([train_x, this_train_x], axis=0)
            train_y = pd.concat([train_y, this_train_y], axis=0)
    return train_x, train_y


if __name__ == '__main__':
    train_x, train_y = load_features()
    feature_names = np.array(train_x.columns)
    train_x.fillna(train_x.mean(), inplace=True)

    train_x_np = train_x.to_numpy()
    train_y_np = np.squeeze(train_y.to_numpy())

    # rank_corr, rank_vals = rank_correlation(train_x_np, train_y_np)
    # ranked_feat_names = feature_names[rank_corr]
    # for i, feat in enumerate(ranked_feat_names[0:20]):
    #     print(f"{feat}: {rank_vals[rank_corr[i]]:.3f}")
    # # results
    # # channel_13__power_gamma: -0.124
    # # channel_10__power_gamma: -0.122
    # # channel_6__power_theta: 0.120
    # # channel_10__skewness: -0.101
    # # channel_15__power_theta: 0.100
    # # channel_7__power_theta: 0.093
    # # channel_9__power_gamma: -0.093
    # # channel_2__root_mean_square: 0.085
    # # channel_2__root_abs_energy: 0.085
    # # channel_2__standard_deviation: 0.083
    # # channel_0__absolute_sum_of_changes: 0.081
    # # channel_2__power_theta: 0.080
    # # channel_11__skewness: -0.079
    # # channel_9__skewness: -0.078
    # # channel_15__standard_deviation: 0.075
    # # channel_15__root_abs_energy: 0.075
    # # channel_15__root_mean_square: 0.075
    # # channel_7__standard_deviation: 0.073
    # # channel_7__root_mean_square: 0.073
    # # channel_7__root_abs_energy: 0.073
    #
    # num_channels = 16
    # channel_abs_rank_sums = np.zeros((num_channels,))
    # for i in range(num_channels):
    #     channel_name = "channel_" + str(i) + "_"
    #     bool_mask = [(channel_name in feat_name) for feat_name in feature_names]
    #     these_rank_vals = np.abs(rank_vals[bool_mask])
    #     channel_abs_rank_sums[i] = np.sum(these_rank_vals)
    # print(channel_abs_rank_sums)
    # print(np.flip(np.argsort(channel_abs_rank_sums)))
    # # results
    # # [0.46228912 0.59031031 0.63162103 0.58676551 0.26726397 0.32537696
    # #  0.62471689 0.58942402 0.32829268 0.43996071 0.53828587 0.50644593
    # #  0.30333964 0.49944856 0.44041579 0.55700525]
    # # [ 2  6  1  7  3 15 10 11 13  0 14  9  8  5 12  4]

    cols_to_drop = []#[col for col in train_x.columns if "channel_2_" not in col]
    train_x_ch2 = train_x.drop(columns=cols_to_drop)
    train_x_np = train_x_ch2.to_numpy()
    train_x, val_x, train_y, val_y = train_test_split(train_x_np, train_y_np, test_size=0.2, random_state=8)
    lr = LogisticRegression(penalty=None)
    lr.fit(train_x, train_y)
    print(train_x.shape)
    print(val_x.shape)

    # print(evaluation_models.get_metrics(lr, train_x, train_y, val_x, val_y))
    #
    # print(evaluation_models.naive_bayes(train_x, train_y, val_x, val_y))
    #
    # print(evaluation_models.ridge_classifier(train_x, train_y, val_x, val_y))
    #
    # print(evaluation_models.adaboost(train_x, train_y, val_x, val_y))

    # train_acc = lr.score(train_x, train_y)
    # val_acc = lr.score(val_x, val_y)
    # train_f1 = f1_score(train_y, lr.predict(train_x))
    # val_f1 = f1_score(val_y, lr.predict(val_x))
    # train_probs = lr.predict_proba(train_x)[:, 1]
    # val_probs = lr.predict_proba(val_x)[:, 1]
    # train_auc = roc_auc_score(train_y, train_probs)
    # val_auc = roc_auc_score(val_y, val_probs)
    # print(f"\nTrain Accuracy: {train_acc}")
    # print(f"Train F1: {train_f1}")
    # print(f"Train AUC: {train_auc}")
    # print(f"\nVal Accuracy: {val_acc}")
    # print(f"Val F1: {val_f1}")
    # print(f"Val AUC: {val_auc}")