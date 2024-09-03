import sys
import os

sys.path.append(os.path.join("..", ".."))
print(sys.path)
from experiment import Exp
import yaml
import os
import torch
from ptflops import get_model_complexity_info

import logging

'''
This script (method) is used to train the model with a particular sensor individually.

The method will train the CV models using three architectures: TinyHAR, Conv-LSTM, and Attend&Discriminate.

:param data_set_index: the index of the sensor to be trained.
'''


def run_train_process_with_data(data_set_index, variant, freq="-1", noise="Y", norm="standardization"):
    # set up the logging
    logging.basicConfig(level=logging.INFO)
    logging.info('Start training process with data' + str(data_set_index) + " " + str(variant))
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = dotdict()
    # TODO change the path as relative path
    path_modifier = ""
    if noise == "Y":
        path_modifier = "/no_noise"
    elif norm == "minmax":
        path_modifier = "/no_norm"
    elif freq != "-1":
        path_modifier = "/no_resamp"
    else:
        path_modifier = ""
    args.to_save_path = r"../../data" + path_modifier + "/Run_logs" + "/" + str(data_set_index) + "-" + variant
    args.freq_save_path = r"../../data" + path_modifier + "/Freq_data"
    args.window_save_path = r"../../data" + path_modifier + "/Sliding_window" + "/" + str(data_set_index) + "-" + variant
    args.root_path = r"../.."
    args.device = data_set_index + "-" + variant
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

    args.data_name = 'realdisp'

    ''' Change this if you wish to train the model with different sampling rate.'''
    if freq is not None and freq != "" and int(freq) != -1:
        args.overwrite_sampling_rate = True
        args.new_sampling_freq = int(freq)
    else:
        args.overwrite_sampling_rate = False
        args.new_sampling_freq = -1

    if noise == "Y":
        args.needs_noise_clean = True
        args.lowcut = 0.5
        args.highcut = 40
    else:
        args.needs_noise_clean = False
        args.lowcut = 0
        args.highcut = 0

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
    if args.overwrite_sampling_rate:
        args.windowsize = int(window_seconds * args.new_sampling_freq)
    else:
        args.windowsize = int(window_seconds * args.sampling_freq)
    args.input_length = args.windowsize
    # input information
    args.c_in = config["num_channels"]

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
    # macs, params = get_model_complexity_info(exp.model, (1, args.input_length, args.c_in), as_strings=False,
    #                                          print_per_layer_stat=True, verbose=False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    exp.train()

    args.model_type = "deepconvlstm"

    exp = Exp(args)
    # macs, params = get_model_complexity_info(exp.model, (1, args.input_length, args.c_in), as_strings=False,
    #                                          print_per_layer_stat=True, verbose=False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    exp.train()

    args.model_type = "attend"

    exp = Exp(args)
    # macs, params = get_model_complexity_info(exp.model, (1, args.input_length, args.c_in), as_strings=False,
    #                                          print_per_layer_stat=True, verbose=False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    exp.train()
