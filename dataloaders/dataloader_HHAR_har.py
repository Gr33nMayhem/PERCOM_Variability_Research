import pandas as pd
import numpy as np
import os
from scipy.signal import butter, lfilter
from dataloaders.dataloader_base import BASE_DATA

HHAR_COLS = ['Index', 'Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt']

HHAR_CV_orig = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
HHAR_CV = [1, 2, 3, 4, 5, 6, 7, 8, 9]

HHARCV_MAP = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
}


# ================================= HARVAR DATASET ============================================
class HHAR_HAR_DATA_loader(BASE_DATA):

    def __init__(self, args):

        # Sensor order: RLA, RUA, BACK, LUA, LLA, RC, RT, LT, LC
        # sensor reading per sensor: ACC, GYR, MAG, ORI
        # Axis per sensor reading ACC, GYR, MAG: X, Y, Z
        # Axis per sensor reading ORI: 1, 2, 3, 4
        self.col_names = HHAR_COLS

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

        # Activities: ‘Biking’, ‘Sitting’, ‘Standing’, ‘Walking’, ‘Stair Up’ and ‘Stair down’.

        self.drop_activities = []

        self.exp_mode = args.exp_mode

        self.split_tag = "sub"

        self.label_map = [
            (0, 'bike'),
            (1, 'sit'),
            (2, 'stand'),
            (3, 'walk'),
            (4, 'stairsup'),
            (5, 'stairsdown'),
        ]

        self.labelToId = list(range(len(self.label_map)))

        self.devices_to_load = args.devices_to_load

        # add 9 if needed
        self.LOCV_keys = []
        for i in HHAR_CV:
            self.LOCV_keys.append([i])

        self.all_keys = HHAR_CV

        self.sub_ids_of_each_sub = {}
        for i in HHAR_CV:
            self.sub_ids_of_each_sub[i] = [i]

        self.labelToId = list(range(len(self.label_map)))
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.data_util = HHARUtils()

        if args.data_name == 'hhar_samsung':
            self.phone_watch = "phone"
        elif args.data_name == 'hhar_nexus':
            self.phone_watch = "phone"
        elif args.data_name == 'hhar_s3':
            self.phone_watch = "phone"
        else:
            self.phone_watch = "watch"

        super(HHAR_HAR_DATA_loader, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        return self.data_util.load_all_the_data_realdisp(root_path, self.device, self.phone_watch)


class HHARUtils:
    def __init__(self):
        self.labelToId = {
            'bike': 0,
            'sit': 1,
            'stand': 2,
            'walk': 3,
            'stairsup': 4,
            'stairsdown': 5
        }
    def load_all_the_data_realdisp(self, root_path, device, phone_watch):
        col_names = HHAR_COLS
        print(" ----------------------- load all the data -------------------")
        # split device string to get the variant and device
        all_data_df = pd.DataFrame()
        if phone_watch == "phone":
            file_name = "/Phones_accelerometer.csv"
        else:
            file_name = "/Watches_accelerometer.csv"

        file_name = root_path + file_name
        for participant in HHARCV_MAP.keys():
            # load using pandas
            data = pd.read_csv(file_name)
            # from data load only where "User" is equal to the participant
            data = data[data["User"] == participant]
            # from data load only where "Device" is device
            data = data[data["Device"] == device]
            # add the subject column
            data["sub_id"] = HHARCV_MAP[participant]
            # remove any rows where "gt" is "null"
            data = data[data["gt"] != "null"]
            # drop na values
            data = data.dropna()
            data["sub"] = data["sub_id"].copy()
            # drop all columns except x, y, z, sub_id, gt
            data = data[["sub_id", "x", "y", "z", "sub", "gt"]].copy()
            # rename gt to activity_id
            data = data.rename(columns={"gt": "activity_id"})
            # replace all activities with their corresponding id
            data["activity_id"] = data["activity_id"].replace(self.labelToId)

            all_data_df = pd.concat([all_data_df, data])

        # Y is the activity_id column
        Y = all_data_df["activity_id"].reset_index(drop=True)
        # X is all the columns except activity_id and sub_id
        X = all_data_df.drop(columns=["activity_id"]).reset_index(drop=True)
        return X, Y
