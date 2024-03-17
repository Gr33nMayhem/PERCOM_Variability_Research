import torch
import sys
import os
import pandas as pd
import numpy as np
import argparse

sys.path.append(os.path.join("..", ".."))

from dataloaders.utils import Normalizer
from dataloaders import dataloader_HARVAR_har, dataloader_REALDISP_har

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset Name')
parser.add_argument('--device_train', type=str, help='Device Name of training')
parser.add_argument('--device_test', type=str, help='Device Name of testing')


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    del xx
    del yy
    del zz
    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def MMD_with_sample(x, y, split_size, iterations, kernal):
    '''Big brain time'''
    all_mmd = []
    for split in range(iterations):
        b1 = np.random.choice(x, split_size, replace=False)
        b2 = np.random.choice(y, split_size, replace=False)
        tensor_a = torch.from_numpy(np.reshape(b1, (len(b1), 1))).to(device)
        tensor_b = torch.from_numpy(np.reshape(b2, (len(b2), 1))).to(device)
        all_mmd.append(MMD(tensor_a, tensor_b, kernel=kernal).item())

    all_mmd = np.array(all_mmd)
    mean_mmd = np.mean(all_mmd)
    std_dev_mmd = np.std(all_mmd)

    return mean_mmd, std_dev_mmd


def load_all_the_data_realdisp(reladisp_device):
    root_path = os.path.join('..', '..', 'data', 'realdisp')
    col_names = ['time_sec', 'time_msec', 'RLA_ACC_X', 'RLA_ACC_Y', 'RLA_ACC_Z', 'RLA_GYR_X', 'RLA_GYR_Y',
                 'RLA_GYR_Z', 'RLA_MAG_X', 'RLA_MAG_Y', 'RLA_MAG_Z', 'RLA_ORI_1', 'RLA_ORI_2', 'RLA_ORI_3',
                 'RLA_ORI_4', 'RUA_ACC_X', 'RUA_ACC_Y', 'RUA_ACC_Z', 'RUA_GYR_X', 'RUA_GYR_Y', 'RUA_GYR_Z',
                 'RUA_MAG_X', 'RUA_MAG_Y', 'RUA_MAG_Z', 'RUA_ORI_1', 'RUA_ORI_2', 'RUA_ORI_3', 'RUA_ORI_4',
                 'BACK_ACC_X', 'BACK_ACC_Y', 'BACK_ACC_Z', 'BACK_GYR_X', 'BACK_GYR_Y', 'BACK_GYR_Z',
                 'BACK_MAG_X', 'BACK_MAG_Y', 'BACK_MAG_Z', 'BACK_ORI_1', 'BACK_ORI_2', 'BACK_ORI_3',
                 'BACK_ORI_4', 'LUA_ACC_X', 'LUA_ACC_Y', 'LUA_ACC_Z', 'LUA_GYR_X', 'LUA_GYR_Y', 'LUA_GYR_Z',
                 'LUA_MAG_X', 'LUA_MAG_Y', 'LUA_MAG_Z', 'LUA_ORI_1', 'LUA_ORI_2', 'LUA_ORI_3', 'LUA_ORI_4',
                 'LLA_ACC_X', 'LLA_ACC_Y', 'LLA_ACC_Z', 'LLA_GYR_X', 'LLA_GYR_Y', 'LLA_GYR_Z', 'LLA_MAG_X',
                 'LLA_MAG_Y', 'LLA_MAG_Z', 'LLA_ORI_1', 'LLA_ORI_2', 'LLA_ORI_3', 'LLA_ORI_4', 'RC_ACC_X',
                 'RC_ACC_Y', 'RC_ACC_Z', 'RC_GYR_X', 'RC_GYR_Y', 'RC_GYR_Z', 'RC_MAG_X', 'RC_MAG_Y',
                 'RC_MAG_Z', 'RC_ORI_1', 'RC_ORI_2', 'RC_ORI_3', 'RC_ORI_4', 'RT_ACC_X', 'RT_ACC_Y',
                 'RT_ACC_Z', 'RT_GYR_X', 'RT_GYR_Y', 'RT_GYR_Z', 'RT_MAG_X', 'RT_MAG_Y', 'RT_MAG_Z',
                 'RT_ORI_1', 'RT_ORI_2', 'RT_ORI_3', 'RT_ORI_4', 'LT_ACC_X', 'LT_ACC_Y', 'LT_ACC_Z',
                 'LT_GYR_X', 'LT_GYR_Y', 'LT_GYR_Z', 'LT_MAG_X', 'LT_MAG_Y', 'LT_MAG_Z', 'LT_ORI_1',
                 'LT_ORI_2', 'LT_ORI_3', 'LT_ORI_4', 'LC_ACC_X', 'LC_ACC_Y', 'LC_ACC_Z', 'LC_GYR_X',
                 'LC_GYR_Y', 'LC_GYR_Z', 'LC_MAG_X', 'LC_MAG_Y', 'LC_MAG_Z', 'LC_ORI_1', 'LC_ORI_2',
                 'LC_ORI_3', 'LC_ORI_4', 'activity_id', 'sub_id']
    all_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    print(" ----------------------- load all the data -------------------")
    # split device string to get the variant and device
    variant_to_use = reladisp_device.split("-")[1]
    device_to_use = reladisp_device.split("-")[0]
    all_data_df = pd.DataFrame()
    for participant in all_keys:
        # from data/realdisp load the .log file of type subject + participant + _ + variant.log
        file_name = root_path + "/subject" + str(participant) + "_" + variant_to_use + ".log"
        # load using pandas
        data = pd.read_csv(file_name, names=col_names, sep='\t')
        # add the subject column
        data["sub_id"] = participant
        # filter out columns from col_names where the column name starts with device + _ and only to use ACC data
        columns_to_keep = [col for col in col_names if
                           col.startswith(device_to_use + "_" + "ACC") or col in ["sub_id", "activity_id"]]
        data = data[columns_to_keep].copy()
        # remove all rows with activity 0
        data = data[data["activity_id"] != 0]
        # create a new column as a copy of sub_id as sub
        data["sub"] = data["sub_id"].copy()
        # make sub_id the first column in data
        data = data[["sub_id"] + [col for col in data.columns if col != "sub_id"]]
        # append the data to all_data_df
        all_data_df = pd.concat([all_data_df, data])

    # Y is the activity_id column
    Y = all_data_df["activity_id"].reset_index(drop=True)
    # X is all the columns except activity_id and sub_id
    X = all_data_df.drop(columns=["activity_id"]).reset_index(drop=True)
    return X, Y


def load_all_the_data_harvar(device):
    print(" ----------------------- load all the data -------------------")

    user_dict = {
        'p001': 1,
        'p002': 2,
        'p003': 3,
        'p004': 4,
        'p007': 5,
        'p008': 6,
        'p009': 7,
        'p013': 8,
    }

    activity_dict = {
        'walking': 0,
        'cooking': 1,
    }

    device_dict = {
        'empatica-left': 'empatica',
        'empatica-right': 'empatica',
        'bluesense-LUA': 'bluesense',
        'bluesense-LWR': 'bluesense',
        'bluesense-RUA': 'bluesense',
        'bluesense-RWR1': 'bluesense',
        'bluesense-RWR2': 'bluesense',
        'bluesense-TRS': 'bluesense',
        'maxim-red': 'maxim',
        'maxim-green': 'maxim',
    }

    root_path = os.path.join('..', '..', 'data', 'harvar')

    data_x = pd.DataFrame(columns=['sub_id', 'x', 'y', 'z'])
    data_y = pd.DataFrame(columns=['activity_id'])
    length_from_first_device = {}
    for key in user_dict.keys():
        for activity in activity_dict.keys():
            data_x, data_y = load_from_csv(data_x, data_y, key, user_dict[key],
                                           activity_dict[activity],
                                           activity, root_path, device_dict[device], device)
            length_from_first_device[key] = data_x.shape[0]
    data_y = data_x.iloc[:, -1]
    data_y = data_y.reset_index(drop=True)
    data_x = data_x.iloc[:, :-1]
    data_x = data_x.reset_index(drop=True)

    X = data_x
    Y = data_y
    return X, Y


""" TODO: Change this when using all labels"""


def extract_labelled_data_only(data_frame, participant_number, root_path):
    label_file = os.path.join(root_path, 'walking/labels', participant_number + '-label-walking' + '.csv')
    label_times = pd.read_csv(label_file, header=0)
    to_concat = []
    for index, row in label_times.iterrows():
        df = data_frame[(data_frame['ts_unix'] >= row['start_timestamp_ms']) & (
                data_frame['ts_unix'] <= row['end_timestamp_ms'])]
        to_concat.append(df)
    temp = pd.concat(to_concat)
    return temp


def load_from_csv(data_x, data_y, participant_id, participant_num, activity_id, activity_name, root_path,
                  device_type, device_id):
    path = os.path.join(root_path, activity_name, device_type)
    temp = pd.read_csv(
        os.path.join(path, participant_id + '-' + device_id + '-' + activity_name + '.csv'))
    if activity_name == 'walking':
        temp = extract_labelled_data_only(temp, participant_id, root_path)
    temp = temp[['Acc_X', 'Acc_Y', 'Acc_Z']]
    temp.columns = ['x', 'y', 'z']
    temp.reset_index(drop=True, inplace=True)
    subj = pd.DataFrame({'sub_id': (np.zeros(temp.shape[0]) + participant_num)})
    temp = subj.join(temp)
    subj = pd.DataFrame({'sub': (np.zeros(temp.shape[0]) + participant_num)})
    temp = temp.join(subj)
    temp = temp.join(pd.DataFrame({'activity_id': np.zeros(temp.shape[0]) + activity_id}))
    data_x = pd.concat([data_x, temp])
    data_y = pd.concat([data_y, pd.DataFrame({'activity_id': (np.zeros(temp.shape[0]) + activity_id)})])
    return data_x, data_y


def normalization(train_vali, test=None):
    train_vali_sensors = train_vali.iloc[:, 1:-1]
    normalizer = Normalizer('standardization')
    normalizer.fit(train_vali_sensors)
    train_vali_sensors = normalizer.normalize(train_vali_sensors)
    train_vali_sensors = pd.concat([train_vali.iloc[:, 0], train_vali_sensors, train_vali.iloc[:, -1]], axis=1)
    if test is None:
        return train_vali_sensors
    else:
        test_sensors = test.iloc[:, 1:-1]
        test_sensors = normalizer.normalize(test_sensors)
        test_sensors = pd.concat([test.iloc[:, 0], test_sensors, test.iloc[:, -1]], axis=1)
        return train_vali_sensors, test_sensors


def run(dataset, device1, device2):
    if os.path.exists(os.path.join('..', '..', 'data', 'mmd', 'mmd_results_' + device1 + '_' + device2 + '.csv')):
        print('mmd results already exist')
        return

    if dataset == 'harvar':
        # harvar
        # iterating through 8 cv
        full_1_x, full_1_y = load_all_the_data_harvar(device1)
        # normalization
        full_1_x = normalization(full_1_x)
        full_2_x, full_2_y = load_all_the_data_harvar(device2)
        # normalization
        full_2_x = normalization(full_2_x)
        num_cv = 8

    else:
        # realdisp
        # iterating through 34 cv
        full_1_x, full_1_y = load_all_the_data_realdisp(device1)
        # normalization
        full_1_x = normalization(full_1_x)
        full_2_x, full_2_y = load_all_the_data_realdisp(device2)
        # normalization
        full_2_x = normalization(full_2_x)
        num_cv = 17

    # create a dataframe to store the mean mmd results on 3 axis
    mean_mmd = pd.DataFrame(columns=['CV', 'Acc_X', 'Acc_Y', 'Acc_Z', 'std_div_x', 'std_div_y', 'std_div_z'])
    for i in range(1, num_cv + 1):
        print('Starting cv', i)
        train = full_1_x[full_1_x['sub_id'] != i]
        test = full_2_x[full_2_x['sub_id'] == i]
        # get only the 'Acc_X', 'Acc_Y', 'Acc_Z' columns as numpy matrix
        train = train.iloc[:, 1:-1].to_numpy()
        test = test.iloc[:, 1:-1].to_numpy()
        # get mmd distance for Acc_X, Acc_Y, Acc_Z
        mean_mmd_x, std_div_x = MMD_with_sample(train[:, 0], test[:, 0], 10000, 500, 'multiscale')
        mean_mmd_y, std_div_y = MMD_with_sample(train[:, 1], test[:, 1], 10000, 500, 'multiscale')
        mean_mmd_z, std_div_z = MMD_with_sample(train[:, 2], test[:, 2], 10000, 500, 'multiscale')
        # store the results in the dataframe
        mean_mmd = pd.concat([mean_mmd, pd.DataFrame({'CV': i, 'Acc_X': mean_mmd_x, 'Acc_Y': mean_mmd_y,
                                                      'Acc_Z': mean_mmd_z, 'std_div_x': std_div_x,
                                                      'std_div_y': std_div_y, 'std_div_z': std_div_z, }, index=[0])],
                             ignore_index=True)
    # save the results in a csv file
    if not os.path.exists(os.path.join('..', '..', 'data', 'mmd')):
        os.makedirs(os.path.join('..', '..', 'data', 'mmd'))
    mean_mmd.to_csv(os.path.join('..', '..', 'data', 'mmd', 'mmd_results_' + device1 + '_' + device2 + '.csv'),
                    index=False)


args = parser.parse_args()
run(args.dataset, args.device_train, args.device_test)
run(args.dataset, args.device_test, args.device_train)
run(args.dataset, args.device_train, args.device_train)
run(args.dataset, args.device_test, args.device_test)
# run('realdisp', 'RLA-ideal', 'RLA-self')
