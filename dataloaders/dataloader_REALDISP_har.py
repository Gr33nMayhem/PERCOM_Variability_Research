import pandas as pd
import numpy as np
import os
from scipy.signal import butter, lfilter
from dataloaders.dataloader_base import BASE_DATA

REALDISP_COLS = ['time_sec', 'time_msec', 'RLA_ACC_X', 'RLA_ACC_Y', 'RLA_ACC_Z', 'RLA_GYR_X', 'RLA_GYR_Y',
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

REALDISP_CV = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


# ================================= HARVAR DATASET ============================================
class REALDISP_HAR_DATA_loader(BASE_DATA):

    def __init__(self, args):

        # Sensor order: RLA, RUA, BACK, LUA, LLA, RC, RT, LT, LC
        # sensor reading per sensor: ACC, GYR, MAG, ORI
        # Axis per sensor reading ACC, GYR, MAG: X, Y, Z
        # Axis per sensor reading ORI: 1, 2, 3, 4
        self.col_names = REALDISP_COLS

        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter = None
        self.sensor_filter = None

        # selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols = None
        # Filtering channels according to the Position
        self.selected_cols = self.col_names
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter,
                                                                              self.col_names[1:], "Sensor Type")
        else:
            self.selected_cols = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter,
                                                                              self.selected_cols, "Sensor Type")

        self.drop_activities = [0]

        self.exp_mode = args.exp_mode

        self.split_tag = "sub"

        self.label_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32, 33]

        self.devices_to_load = args.devices_to_load

        # add 9 if needed
        self.LOCV_keys = []
        for i in REALDISP_CV:
            self.LOCV_keys.append([i])

        self.all_keys = REALDISP_CV

        self.sub_ids_of_each_sub = {}
        for i in REALDISP_CV:
            self.sub_ids_of_each_sub[i] = [i]

        self.labelToId = list(range(len(self.label_map)))
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.data_util = REALDISPUtils()

        super(REALDISP_HAR_DATA_loader, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        return self.data_util.load_all_the_data_realdisp(root_path, self.device, self.all_keys)


class REALDISPUtils:
    def load_all_the_data_realdisp(self, root_path, reladisp_device, participants):
        col_names = REALDISP_COLS
        all_keys = participants
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
