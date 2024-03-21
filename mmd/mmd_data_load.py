import os
import pandas as pd
import numpy as np


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
