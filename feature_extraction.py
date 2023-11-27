from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
from tsfresh.feature_extraction.feature_calculators import abs_energy
from tsfresh.feature_extraction import extract_features


Fs = 399.6097561
window_size = 1200  # 3s
window_overlap = 400  # 1s
processing_chunk_num_windows = 30


def segment_signal_multichannel(signals, window_size, overlap, Fs, window_id_start=0, window_start_ind=0):
    """ expects signals to be number of channels x number of eeg samples
    """
    window_id = window_id_start
    num_channels = signals.shape[0]
    signal_length = signals.shape[1]
    t_axis = np.arange(window_start_ind/Fs, (window_start_ind+signal_length) / Fs, 1 / Fs)
    upper_time = 1200  # 20 min
    signal_cols = ["channel_" + str(i) for i in range(num_channels)]
    result_df = pd.DataFrame(columns=["window_id", "time"] + signal_cols)
    signals = np.transpose(signals)

    window_start_ind = 0
    while window_start_ind < signal_length:
        window_end_ind = np.min([signal_length, window_start_ind + window_size])

        this_time = upper_time * np.ones((window_size,))
        this_signal = np.zeros((window_size, num_channels))
        this_time[:window_end_ind - window_start_ind] = t_axis[window_start_ind:window_end_ind]
        this_signal[:window_end_ind - window_start_ind, :] = signals[window_start_ind:window_end_ind, :]

        this_df = pd.DataFrame({"window_id": window_id, "time": this_time})
        for i, channel_name in enumerate(signal_cols):
            this_df[channel_name] = this_signal[:, i]
        if result_df.shape[0] == 0:
            result_df = this_df
        else:
            result_df = pd.concat([result_df, this_df])

        window_start_ind += window_size - overlap
        window_id += 1

        if window_end_ind >= signal_length:
            break
    return result_df


def get_data_key_from_filename(filename):
    filename = filename.split("/")
    filename = filename[-1].split(".")
    filename = filename[0].split("_")
    filename = filename[2:]
    filename[-1] = filename[-1].lstrip("0")
    return "_".join(filename)


def generate_and_save_windowed_data():
    root_folder = "/Volumes/LACIE SHARE/seizure-prediction"
    output_folder = "/Volumes/LACIE SHARE/seizure-prediction-outputs/windows_3sduration_1soverlap"
    train_files_undersamp = [f.strip() for f in open("./data/train_undersampled.txt").readlines()]
    for i, filename in enumerate(train_files_undersamp):
        print(f"Segmenting file {i}/{len(train_files_undersamp)}: {filename.split('/')[-1]}")
        filepath = os.path.join(root_folder, filename)
        mat_data = loadmat(filepath)
        signals = np.array(mat_data[get_data_key_from_filename(filename)][0, 0][0])
        outfilename = filename.split("/")[-1]
        outfilename = outfilename.split(".")[0] + ".csv"
        outpath = os.path.join(output_folder, outfilename)
        # clear output file if it already exists
        if os.path.exists(outpath):
            os.remove(outpath)

        window_start_ind = 0
        window_end_ind = window_size + (window_size - window_overlap) * (processing_chunk_num_windows - 1)
        window_id_start = 0
        while window_start_ind < signals.shape[1]:
            chunk_df = segment_signal_multichannel(signals[:, window_start_ind:window_end_ind], window_size, window_overlap,
                                                   Fs, window_id_start, window_start_ind)

            if window_start_ind == 0:
                # write header
                chunk_df.to_csv(outpath, index=False, mode='a')
            else:
                # exclude  header
                chunk_df.to_csv(outpath, index=False, mode='a', header=False)
            # print(chunk_df['window_id'].max())

            window_start_ind = window_end_ind - window_overlap

            window_end_ind = window_start_ind + window_size + (window_size - window_overlap) * (
                        processing_chunk_num_windows - 1)

            window_id_start += processing_chunk_num_windows

        # results = pd.read_csv(outpath)
        # print(results.value_counts(subset='window_id').max())
    print("Done")


def root_abs_energy(x):
    return np.sqrt(abs_energy(x))


def get_powers(x):
    freq = np.fft.fftfreq(x.shape[0], d=1 / Fs)
    power_spectrum = np.abs(np.fft.fft(x)) ** 2
    results = dict()
    results['power_delta'] = np.mean(power_spectrum[(freq >= 1) & (freq < 4)])
    results['power_theta'] = np.mean(power_spectrum[(freq >= 4) & (freq < 8)])
    results['power_alpha'] = np.mean(power_spectrum[(freq >= 8) & (freq < 14)])
    results['power_beta'] = np.mean(power_spectrum[(freq >= 14) & (freq < 30)])
    results['power_gamma'] = np.mean(power_spectrum[(freq >= 30)])
    return results


def get_custom_feat(x):
    results = get_powers(x)
    results['root_abs_energy'] = np.sqrt(abs_energy(x))
    return results


def get_custom_feat_multichannel(x):
    """Assumes x is number of timepoints x number of channels
    """
    results = dict()
    for channel_num in range(x.shape[1]):
        these_results = get_powers(x[:, channel_num])
        for key, value in these_results.items():
            results['channel_' + str(channel_num) + '__' + key] = value
        results['channel_' + str(channel_num) + '__root_abs_energy'] = np.sqrt(abs_energy(x[:, channel_num]))
    return results


fc_parameters = {
    "mean": None,
    "standard_deviation": None,
    "absolute_sum_of_changes": None,
    "minimum": None,
    "maximum": None,
    "skewness": None,
    "kurtosis": None,
    "root_mean_square": None
}


def extract_feat_per_window(window_df, sample_col='window_id'):
    ts_fresh_feat = extract_features(window_df, default_fc_parameters=fc_parameters, column_id='window_id', column_sort='time', n_jobs=5)
    cust_results = []
    for i in range(window_df[sample_col].max()+1):
        this_sig = window_df[window_df[sample_col] == i]
        this_sig = this_sig.drop(columns=['window_id', 'time']).to_numpy()
        cust_results.append(get_custom_feat_multichannel(this_sig))
    all_feat = pd.concat([ts_fresh_feat, pd.DataFrame(cust_results)], axis=1)
    return all_feat


train_undersampled_windowed_data_folder = "/Volumes/LACIE SHARE/seizure-prediction-outputs/windows_3sduration_1soverlap"
train_undersampled_feat_output_folder = "/Volumes/LACIE SHARE/seizure-prediction-outputs/windows_3sduration_1soverlap_features"
def extract_feat_train_undersampled():
    files_to_process = [filename for filename in os.listdir(train_undersampled_windowed_data_folder) if "._" not in filename]
    for i, filename in enumerate(files_to_process):
        print(f"Extracting features for file {i}/{len(files_to_process)}: {filename}")
        data = pd.read_csv(os.path.join(train_undersampled_windowed_data_folder, filename))
        feat = extract_feat_per_window(data)
        outfilename = filename.split(".")[0] + "_features.csv"
        feat.to_csv(os.path.join(train_undersampled_feat_output_folder, outfilename))
    print("Done")


if __name__ == '__main__':
    # generate_and_save_windowed_data()
    extract_feat_train_undersampled()
