import sys
import os

sys.path.append(os.path.join("..", ".."))
print(sys.path)
from experiment import Exp
import yaml
import os
import torch
from ptflops import get_model_complexity_info

'''
This script (method) is used to train the model with a particular sensor individually.

The method will train the CV models using three architectures: TinyHAR, Conv-LSTM, and Attend&Discriminate.

:param data_set_index: the index of the sensor to be trained.
'''
devices = ['bluesense-RWR1',
           'bluesense-RWR2']


def run_test_process_with_data(data_set_index):
    print("started running train process...")

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = dotdict()
    args.devices_to_load = devices
    # TODO change the path as relative path
    args.to_save_path = r"../../data/Run_logs" + "/" + str(devices[data_set_index])
    args.freq_save_path = r"../../data/Freq_data"
    args.window_save_path = r"../../data/Sliding_window"
    args.root_path = r"../.."

    args.drop_transition = False
    args.datanorm_type = "standardization"  # None ,"standardization", "minmax"
    args.filter_scaling_factor = 1
    args.batch_size = 256
    args.shuffle = True
    args.drop_last = False
    args.train_vali_quote = 0.90
    # training setting
    args.train_epochs = 150
    args.learning_rate = 0.001
    args.learning_rate_patience = 7
    args.learning_rate_factor = 0.1
    args.early_stop_patience = 15

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False

    args.optimizer = "Adam"
    args.criterion = "CrossEntropy"
    args.seed = 1
    args.data_name = 'harvar'
    args.wavelet_filtering = False
    args.wavelet_filtering_regularization = False
    args.wavelet_filtering_finetuning = False
    args.wavelet_filtering_finetuning_percent = 0.5
    args.wavelet_filtering_learnable = False
    args.wavelet_filtering_layernorm = False
    args.regulatization_tradeoff = 0
    args.number_wavelet_filtering = 12
    args.difference = False
    args.filtering = False
    args.magnitude = False
    args.weighted_sampler = True
    args.pos_select = None
    args.sensor_select = None
    args.representation_type = "time"
    args.exp_mode = "LOCV"

    config_file = open('../../configs/data.yaml', mode='r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = data_config[args.data_name]

    args.root_path = os.path.join(args.root_path, config["filename"])
    args.sampling_freq = config["sampling_freq"]
    args.num_classes = config["num_classes"]
    window_seconds = config["window_seconds"]
    args.windowsize = int(window_seconds * args.sampling_freq)
    args.input_length = args.windowsize
    # input information
    args.c_in = config["num_channels"]

    args.train_variant = data_set_index
    args.test_variant = data_set_index
    args.variation_count = 1

    if args.difference:
        args.c_in = args.c_in * 2

    if args.wavelet_filtering:

        if args.windowsize % 2 == 1:
            N_ds = int(torch.log2(torch.tensor(args.windowsize - 1)).floor()) - 2
        else:
            N_ds = int(torch.log2(torch.tensor(args.windowsize)).floor()) - 2

        args.f_in = args.number_wavelet_filtering * N_ds + 1
    else:
        args.f_in = 1

    args.model_type = "tinyhar"

    args.cross_channel_interaction_type = "attn"
    args.cross_channel_aggregation_type = "FC"
    args.temporal_info_interaction_type = "lstm"
    args.temporal_info_aggregation_type = "tnaive"
    exp = Exp(args)
    exp.prediction_test()

    args.model_type = "deepconvlstm"

    exp = Exp(args)
    exp.prediction_test()

    args.model_type = "attend"

    exp = Exp(args)
    exp.prediction_test()
