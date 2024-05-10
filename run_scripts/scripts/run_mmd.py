import torch
import sys
import os
import pandas as pd
import numpy as np
import argparse

sys.path.append(os.path.join("..", ".."))
from mmd.mmd import MMD_with_sample
from dataloaders.dataloader_HARVAR_har import HARVARUtils
from dataloaders.dataloader_HARVAR_har import HARVAR_CV
from dataloaders.dataloader_REALDISP_har import REALDISPUtils
from dataloaders.dataloader_REALDISP_har import REALDISP_CV

from dataloaders.utils import Normalizer
from configs.config_consts import REALDISP_CV, HARVAR_CV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset Name')
parser.add_argument('--device_train', type=str, help='Device Name of training')
parser.add_argument('--device_test', type=str, help='Device Name of testing')


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
    # if os.path.exists(os.path.join('..', '..', 'data', 'mmd', 'mmd_results_' + device1 + '_' + device2 + '.csv')):
    #     print('mmd results already exist')
    #     return

    if dataset == 'harvar':
        data_utils = HARVARUtils()
        # harvar
        # iterating through 8 cv
        full_1_x, full_1_y = data_utils.load_all_the_data_harvar(device1, HARVAR_CV, True)
        # normalization
        full_1_x = normalization(full_1_x)
        full_2_x, full_2_y = data_utils.load_all_the_data_harvar(device2, HARVAR_CV, True)
        # normalization
        full_2_x = normalization(full_2_x)
        participants = HARVAR_CV

    else:
        data_utils = REALDISPUtils()
        root_path = os.path.join('..', '..', 'data', 'realdisp')
        # realdisp
        # iterating through 34 cv
        full_1_x, full_1_y = data_utils.load_all_the_data_realdisp(root_path, device1, REALDISP_CV)
        # normalization
        full_1_x = normalization(full_1_x)
        full_2_x, full_2_y = data_utils.load_all_the_data_realdisp(root_path, device2, REALDISP_CV)
        # normalization
        full_2_x = normalization(full_2_x)
        participants = REALDISP_CV

    # bandwidth ranges
    bandwidth_range = [0.2, 0.5, 0.9, 1.3, 1.5, 1.6]

    # create a dataframe to store the mean mmd results on 3 axis
    mean_mmd = pd.DataFrame(columns=['CV', 'Acc_X', 'Acc_Y', 'Acc_Z', 'std_div_x', 'std_div_y', 'std_div_z'])
    for i in participants:
        print('Starting cv', i)
        train = full_1_x[full_1_x['sub_id'] != i]
        test = full_2_x[full_2_x['sub_id'] == i]
        # get only the 'Acc_X', 'Acc_Y', 'Acc_Z' columns as numpy matrix
        train = train.iloc[:, 1:-1].to_numpy()
        test = test.iloc[:, 1:-1].to_numpy()
        # get mmd distance for Acc_X, Acc_Y, Acc_Z
        mean_mmd_x, std_div_x = MMD_with_sample(train[:, 0], test[:, 0], 100, 50000, 'multiscale', bandwidth_range)
        mean_mmd_y, std_div_y = MMD_with_sample(train[:, 1], test[:, 1], 100, 50000, 'multiscale', bandwidth_range)
        mean_mmd_z, std_div_z = MMD_with_sample(train[:, 2], test[:, 2], 100, 50000, 'multiscale', bandwidth_range)
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
