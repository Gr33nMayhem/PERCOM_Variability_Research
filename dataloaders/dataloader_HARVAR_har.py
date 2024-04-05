import pandas as pd
import numpy as np
import os
from scipy.signal import butter, lfilter
from dataloaders.dataloader_base import BASE_DATA
from configs.config_consts import HARVAR_CV

HARVAR_CV = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]


# ================================= HARVAR DATASET ============================================
class HARVAR_HAR_DATA_loader(BASE_DATA):

    def __init__(self, args):

        self.col_names = ['sub_id', 'device', 'x', 'y', 'z', 'activity_id']

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

        self.drop_activities = []
        self.exp_mode = args.exp_mode

        self.split_tag = "sub"
        self.only_walking = args.only_walking
        if self.only_walking:
            self.label_map = [
                (0, 'walking'),
                (1, 'not walking'),
            ]
        else:
            self.label_map = [
                (0, 'Walking 2mph'),
                (1, 'Walking 2.5mph'),
                (2, 'Walking 3mph'),
                (3, 'Walking 3.5mph'),
                (4, 'Walking 4mph'),
                (5, 'Washing Hands'),
                (6, 'Bite'),
                (7, 'Drinking'),
            ]

        self.devices_to_load = args.devices_to_load

        # add 9 if needed
        self.LOCV_keys = []
        for i in HARVAR_CV:
            self.LOCV_keys.append([i])

        self.all_keys = HARVAR_CV

        self.sub_ids_of_each_sub = {}
        for i in HARVAR_CV:
            self.sub_ids_of_each_sub[i] = [i]

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.data_util = HARVARUtils()

        super(HARVAR_HAR_DATA_loader, self).__init__(args)

    def load_all_the_data(self, root_path):
        return self.data_util.load_all_the_data_harvar(self.device, HARVAR_CV, self.only_walking)


class HARVARUtils:
    def load_all_the_data_harvar(self, device, participants, only_walking):
        print(" ----------------------- load all the data -------------------")

        if only_walking:
            activity_dict = {
                'walking': {'walking': 0},
                'cooking': {'not_walking': 1},
            }

        else:
            activity_dict = {
                'walking': {'Walking 2mph': 0, 'Walking 2.5mph': 1, 'Walking 3mph': 2, 'Walking 3.5mph': 3,
                            'Walking 4mph': 4},
                'cooking': {'Washing Hands': 5, 'Bite': 6, 'Drinking': 7},
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
        for key in participants:
            for directory in activity_dict.keys():
                data_x, data_y = self.load_from_csv(data_x, data_y, "p{:03d}".format(key), key,
                                                    activity_dict[directory],
                                                    directory, root_path, device_dict[device], device)
                length_from_first_device[key] = data_x.shape[0]
        data_y = data_x.iloc[:, -1]
        data_y = data_y.reset_index(drop=True)
        data_x = data_x.iloc[:, :-1]
        data_x = data_x.reset_index(drop=True)

        X = data_x
        Y = data_y
        return X, Y

    def extract_labelled_data_only(self, data_frame, participant_number, root_path, activity):
        if activity == 'walking':
            label_file = os.path.join(root_path, 'walking/labels', participant_number + '-label-walking' + '.csv')
            label_times = pd.read_csv(label_file, header=0)
            to_concat = []
            for index, row in label_times.iterrows():
                df = data_frame[(data_frame['ts_unix'] >= row['start_timestamp_ms']) & (
                        data_frame['ts_unix'] <= row['end_timestamp_ms'])]
                to_concat.append(df)
            temp = pd.concat(to_concat)
        elif activity == 'not_walking':
            # Dont do anything, as we want all of the cooking dataset
            temp = data_frame
        else:
            # TODO: Add other activities and their labels
            temp = data_frame

        return temp

    def load_from_csv(self, data_x, data_y, participant_id, participant_num, activities, directory_name, root_path,
                      device_type, device_id):
        path = os.path.join(root_path, directory_name, device_type)
        temp = pd.read_csv(
            os.path.join(path, participant_id + '-' + device_id + '-' + directory_name + '.csv'))
        for activity in activities.keys():
            temp = self.extract_labelled_data_only(temp, participant_id, root_path, activity)
            temp = temp[['Acc_X', 'Acc_Y', 'Acc_Z']]
            temp.columns = ['x', 'y', 'z']
            temp.reset_index(drop=True, inplace=True)
            subj = pd.DataFrame({'sub_id': (np.zeros(temp.shape[0]) + participant_num)})
            temp = subj.join(temp)
            subj = pd.DataFrame({'sub': (np.zeros(temp.shape[0]) + participant_num)})
            temp = temp.join(subj)
            temp = temp.join(pd.DataFrame({'activity_id': np.zeros(temp.shape[0]) + activities[activity]}))
            data_x = pd.concat([data_x, temp])
            data_y = pd.concat(
                [data_y, pd.DataFrame({'activity_id': (np.zeros(temp.shape[0]) + activities[activity])})])
        return data_x, data_y
