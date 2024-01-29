import torch
import sys
import os
import pandas as pd
import numpy as np
import argparse

sys.path.append(os.path.join("..", ".."))

from dataloaders.utils import Normalizer
from dataloaders import dataloader_HARVAR_har

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='Enter a sensor name as --device_train <device name> --device_test \n ' +
                'empatica-left: empatica\n' +
                'empatica-right: empatica\n' +
                'bluesense-LUA: bluesense\n' +
                'bluesense-LWR: bluesense\n' +
                'bluesense-RUA: bluesense\n' +
                'bluesense-RWR1: bluesense\n' +
                'bluesense-RWR2: bluesense\n' +
                'bluesense-TRS: bluesense\n' +
                'maxim-red: maxim\n' +
                'maxim-green: maxim')
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


def load_all_the_data(device):
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


def run(device1, device2):
    if os.path.exists(os.path.join('..', '..', 'data', 'mmd', 'mmd_results_' + device1 + '_' + device2 + '.csv')):
        print('mmd results already exist')
        return
    # sample compare between bluesense-RWR1 and bluesense-RWR2
    # iterating through 8 cv
    full_1_x, full_1_y = load_all_the_data(device1)
    normalization(full_1_x)
    full_2_x, full_2_y = load_all_the_data(device2)
    normalization(full_2_x)
    # create a dataframe to store the mean mmd results on 3 axis
    mean_mmd = pd.DataFrame(columns=['CV', 'Acc_X', 'Acc_Y', 'Acc_Z', 'std_div_x', 'std_div_y', 'std_div_z'])
    for i in range(1, 9):
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
run(args.device_train, args.device_test)
run(args.device_test, args.device_train)
run(args.device_train, args.device_train)
run(args.device_test, args.device_test)
