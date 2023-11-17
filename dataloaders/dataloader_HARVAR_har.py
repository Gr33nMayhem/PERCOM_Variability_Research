import pandas as pd
import numpy as np
import os
from scipy.signal import butter, lfilter
from dataloaders.dataloader_base import BASE_DATA


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

        self.label_map = [
            (0, 'walking'),
            (1, 'not walking'),
        ]

        self.devices_to_load = args.devices_to_load

        # add 9 if needed
        self.LOCV_keys = [[1], [2], [3], [4], [5], [6], [7], [8]]
        self.all_keys = [1, 2, 3, 4, 5, 6, 7, 8]
        self.sub_ids_of_each_sub = {1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8]}

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(HARVAR_HAR_DATA_loader, self).__init__(args)

    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")
        cooking_data_path = '../../data/harvar/cooking'
        walking_data_path = '../../data/harvar/walking'

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
        X = []
        Y = []
        lengths_for_activities = {}
        for device in self.devices_to_load:
            data_x = pd.DataFrame(columns=['sub_id', 'x', 'y', 'z'])
            data_y = pd.DataFrame(columns=['activity_id'])
            length_from_first_device = {}
            for activity in activity_dict.keys():
                for key in user_dict.keys():
                    # check if the index of the device is the first
                    if self.devices_to_load.index(device) == 0:
                        data_x, data_y = self.load_from_csv(data_x, data_y, key, user_dict[key],
                                                            activity_dict[activity],
                                                            activity, root_path, device_dict[device], device, 0)
                        length_from_first_device[key] = data_x.shape[0]
                    else:
                        data_x, data_y = self.load_from_csv(data_x, data_y, key, user_dict[key],
                                                            activity_dict[activity],
                                                            activity, root_path, device_dict[device], device,
                                                            lengths_for_activities[activity][key])
                # save the length of the first device per activity, so that it can be used for cropping
                lengths_for_activities[activity] = length_from_first_device
            data_y = data_x.iloc[:, -1]
            data_y = data_y.reset_index(drop=True)
            data_x = data_x.iloc[:, :-1]
            data_x = data_x.reset_index(drop=True)

            X.append(data_x)
            Y.append(data_y)
        return X, Y

    """ TODO: Change this when using all labels"""

    def extract_labelled_data_only(self, data_frame, participant_number, root_path):
        label_file = os.path.join(root_path, 'walking/labels', participant_number + '-label-walking' + '.csv')
        label_times = pd.read_csv(label_file, header=0)
        to_concat = []
        for index, row in label_times.iterrows():
            df = data_frame[(data_frame['ts_unix'] >= row['start_timestamp_ms']) & (
                    data_frame['ts_unix'] <= row['end_timestamp_ms'])]
            to_concat.append(df)
        temp = pd.concat(to_concat)
        return temp

    def load_from_csv(self, data_x, data_y, participant_id, participant_num, activity_id, activity_name, root_path,
                      device_type, device_id, need_cropping=0):
        empatica_path = os.path.join(root_path, activity_name, device_type)
        temp = pd.read_csv(
            os.path.join(empatica_path, participant_id + '-' + device_id + '-' + activity_name + '.csv'))
        if activity_name == 'walking':
            temp = self.extract_labelled_data_only(temp, participant_id, root_path)
        temp = temp[['Acc_X', 'Acc_Y', 'Acc_Z']]
        temp.columns = ['x', 'y', 'z']
        if need_cropping != 0:
            temp = temp[:need_cropping]
        temp.reset_index(drop=True, inplace=True)
        subj = pd.DataFrame({'sub_id': (np.zeros(temp.shape[0]) + participant_num)})
        temp = subj.join(temp)
        subj = pd.DataFrame({'sub': (np.zeros(temp.shape[0]) + participant_num)})
        temp = temp.join(subj)
        temp = temp.join(pd.DataFrame({'activity_id': np.zeros(temp.shape[0]) + activity_id}))
        data_x = pd.concat([data_x, temp])
        data_y = pd.concat([data_y, pd.DataFrame({'activity_id': (np.zeros(temp.shape[0]) + activity_id)})])
        return data_x, data_y
