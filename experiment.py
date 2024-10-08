import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
import time
from dataloaders import data_dict, data_set
from sklearn.metrics import confusion_matrix
import yaml
# import models
from models.model_builder import model_builder
import warnings

warnings.filterwarnings("ignore")
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from utils import EarlyStopping, adjust_learning_rate_class, mixup_data, MixUpLoss
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import random
import os
import logging

logging.basicConfig(level=logging.INFO)


class Exp(object):
    def __init__(self, args):
        self.args = args

        # set the device
        self.device = self.acquire_device()

        self.optimizer_dict = {"Adam": optim.Adam}
        self.criterion_dict = {"MSE": nn.MSELoss, "CrossEntropy": nn.CrossEntropyLoss}

        self.model = self.build_model().to(self.device)
        logging.info("Done!")
        self.model_size = np.sum([para.numel() for para in self.model.parameters() if para.requires_grad])
        logging.info("Parameter :{}".format(self.model_size))

        logging.info("Set the seed as : {}".format(self.args.seed))

    def acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            logging.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            logging.info('Use CPU')
        return device

    def build_model(self):
        model = model_builder(self.args)
        return model.double()

    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError
        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError
        criterion = self.criterion_dict[self.args.criterion]()
        return criterion

    def _get_data(self, data, flag="train", weighted_sampler=False):
        if flag == 'train':
            shuffle_flag = True
        else:
            shuffle_flag = False

        data = data_set(self.args, data, flag)
        if weighted_sampler and flag == 'train':

            sampler = WeightedRandomSampler(
                data.act_weights, len(data.act_weights)
            )

            data_loader = DataLoader(data,
                                     batch_size=self.args.batch_size,
                                     # shuffle      =  shuffle_flag,
                                     num_workers=6,
                                     sampler=sampler,
                                     drop_last=False)
        else:
            data_loader = DataLoader(data,
                                     batch_size=self.args.batch_size,
                                     shuffle=shuffle_flag,
                                     num_workers=6,
                                     drop_last=False)
        return data_loader

    def get_setting_name(self):
        if self.args.model_type == "deepconvlstm":
            config_file = open('../../configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["deepconvlstm"]
            setting = "deepconvlstm_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_lstmfilter_{}_Regu_{}_wavelearnble_{}".format(
                self.args.data_name,
                self.args.seed,
                self.args.windowsize,
                self.args.wavelet_filtering,
                self.args.filter_scaling_factor,
                config["nb_filters"],
                config["nb_units_lstm"],
                self.args.wavelet_filtering_regularization,
                self.args.wavelet_filtering_learnable)
            return setting

        if self.args.model_type == "deepconvlstm_attn":
            config_file = open('../../configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["deepconvlstm_attn"]
            setting = "deepconvlstm_attn_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_lstmfilter_{}_Regu_{}_wavelearnble_{}".format(
                self.args.data_name,
                self.args.seed,
                self.args.windowsize,
                self.args.wavelet_filtering,
                self.args.filter_scaling_factor,
                config["nb_filters"],
                config["nb_units_lstm"],
                self.args.wavelet_filtering_regularization,
                self.args.wavelet_filtering_learnable)
            return setting

        if self.args.model_type == "mcnn":
            config_file = open('../../configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["mcnn"]
            setting = "mcnn_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_Regu_{}_wavelearnble_{}".format(
                self.args.data_name,
                self.args.seed,
                self.args.windowsize,
                self.args.wavelet_filtering,
                self.args.filter_scaling_factor,
                config["nb_filters"],
                self.args.wavelet_filtering_regularization,
                self.args.wavelet_filtering_learnable)
            return setting

        elif self.args.model_type == "attend":
            config_file = open('../../configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["attend"]
            setting = "attend_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_grufilter_{}_Regu_{}_wavelearnble_{}".format(
                self.args.data_name,
                self.args.seed,
                self.args.windowsize,
                self.args.wavelet_filtering,
                self.args.filter_scaling_factor,
                config["filter_num"],
                config["hidden_dim"],
                self.args.wavelet_filtering_regularization,
                self.args.wavelet_filtering_learnable)
            return setting
        elif self.args.model_type == "sahar":
            config_file = open('../../configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["sahar"]
            setting = "sahar_data_{}_seed_{}_windowsize_{}_waveFilter_{}_Fscaling_{}_cvfilter_{}_grufilter_{}_Regu_{}_wavelearnble_{}".format(
                self.args.data_name,
                self.args.seed,
                self.args.windowsize,
                self.args.wavelet_filtering,
                self.args.filter_scaling_factor,
                config["nb_filters"],
                None,
                self.args.wavelet_filtering_regularization,
                self.args.wavelet_filtering_learnable)
            return setting
        elif self.args.model_type == "tinyhar":
            config_file = open('../../configs/model.yaml', mode='r')
            config = yaml.load(config_file, Loader=yaml.FullLoader)["tinyhar"]
            setting = "tinyhar_data_{}_seed_{}_windowsize_{}_cvfilter_{}_CI_{}_CA_{}_TI_{}_TA_{}".format(
                self.args.data_name,
                self.args.seed,
                self.args.windowsize,
                config["filter_num"],
                self.args.cross_channel_interaction_type,
                self.args.cross_channel_aggregation_type,
                self.args.temporal_info_interaction_type,
                self.args.temporal_info_aggregation_type)
            return setting
        else:
            raise NotImplementedError

    def update_gamma(self):
        for n, parameter in self.model.named_parameters():
            if "gamma" in n:
                parameter.grad.data.add_(self.args.regulatization_tradeoff * torch.sign(parameter.data))  # L1

    def train(self):

        setting = self.get_setting_name()

        path = os.path.join(self.args.to_save_path, 'logs/' + setting)
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        score_log_file_name = os.path.join(self.path, "score.txt")

        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        # load the data
        dataset = data_dict[self.args.data_name](self.args)

        logging.info("================ {} Mode ====================".format(dataset.exp_mode))
        logging.info("================ {} CV ======================".format(dataset.num_of_cv))

        num_of_cv = dataset.num_of_cv

        for iter in range(num_of_cv):

            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            g = torch.Generator()
            g.manual_seed(self.args.seed)
            torch.backends.cudnn.benchmark = False
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

            logging.info("================ the {} th CV Experiment ================ ".format(iter))

            dataset.update_train_val_test_keys()

            cv_path = os.path.join(self.path, "cv_{}".format(iter))
            # get the loader of train val test

            train_loader = self._get_data(dataset, flag='train', weighted_sampler=self.args.weighted_sampler)
            val_loader = self._get_data(dataset, flag='vali', weighted_sampler=self.args.weighted_sampler)
            test_loader = self._get_data(dataset, flag='test', weighted_sampler=self.args.weighted_sampler)
            # class_weights=torch.tensor(dataset.act_weights,dtype=torch.double).to(self.device)
            train_steps = len(train_loader)

            if not os.path.exists(cv_path):
                os.makedirs(cv_path)
                skip_train = False
                skip_finetuning = False
            else:
                file_in_folder = os.listdir(cv_path)
                if 'final_best_vali.pth' in file_in_folder:
                    skip_train = True
                else:
                    skip_train = False

                if 'final_finetuned_best_vali.pth' in file_in_folder:
                    skip_finetuning = True
                else:
                    skip_finetuning = False

            epoch_log_file_name = os.path.join(cv_path, "epoch_log.txt")

            if skip_train:
                logging.info("================Skip the {} CV Experiment================".format(iter))
            else:

                if os.path.exists(epoch_log_file_name):
                    os.remove(epoch_log_file_name)

                epoch_log = open(epoch_log_file_name, "a")
                score_log = open(score_log_file_name, "a")

                logging.info("================ Build the model ================ ")
                if self.args.mixup:
                    logging.info(" Using Mixup Training")
                self.model = self.build_model().to(self.device)

                early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
                learning_rate_adapter = adjust_learning_rate_class(self.args, True)
                model_optim = self._select_optimizer()

                # if self.args.weighted == True:
                #    criterion =  nn.CrossEntropyLoss(reduction="mean",weight=class_weights).to(self.device)#self._select_criterion()
                # else:
                #    criterion =  nn.CrossEntropyLoss(reduction="mean").to(self.device)#self._select_criterion()
                criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

                for epoch in range(self.args.train_epochs):

                    train_loss = []
                    self.model.train()
                    epoch_time = time.time()

                    for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):
                        # if "cross" in self.args.model_type:
                        #    batch_x1 = batch_x1.double().to(self.device)
                        #    batch_x2 = batch_x2.double().to(self.device)
                        #    batch_y = batch_y.long().to(self.device)
                        #    # model prediction
                        #    if self.args.output_attention:
                        #        outputs = self.model(batch_x1,batch_x2)[0]
                        #    else:
                        #        outputs = self.model(batch_x1,batch_x2)
                        # else:
                        batch_x1 = batch_x1.double().to(self.device)  # --
                        batch_y = batch_y.long().to(self.device)  # --

                        #    if self.args.mixup:
                        #        batch_x1, batch_y = mixup_data(batch_x1, batch_y, self.args.alpha)

                        #    # model prediction
                        #    if self.args.output_attention:
                        #        outputs = self.model(batch_x1)[0]
                        #    else:
                        outputs = self.model(batch_x1)  # --

                        # if self.args.mixup:
                        #    criterion = MixUpLoss(criterion)
                        #    loss = criterion(outputs, batch_y)
                        # else:
                        loss = criterion(outputs, batch_y)  # --

                        if self.args.wavelet_filtering and self.args.wavelet_filtering_regularization:
                            reg_loss = 0
                            for name, parameter in self.model.named_parameters():
                                if "gamma" in name:
                                    reg_loss += torch.sum(torch.abs(parameter))

                            loss = loss + self.args.regulatization_tradeoff * reg_loss

                        train_loss.append(loss.item())

                        model_optim.zero_grad()
                        loss.backward()
                        model_optim.step()

                    logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    epoch_log.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    epoch_log.write("\n")

                    train_loss = np.average(train_loss)
                    vali_time = time.time()
                    vali_loss, vali_acc, vali_f_w, vali_f_macro, vali_f_micro = self.validation(self.model, val_loader,
                                                                                                criterion)
                    logging.info("Validation for Epoch: {} cost time: {}".format(epoch + 1, time.time() - vali_time))

                    logging.info(
                        "VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} ".format(
                            epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))

                    epoch_log.write(
                        "VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} \n".format(
                            epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))

                    early_stopping(vali_loss, self.model, cv_path, vali_f_macro, vali_f_w, epoch_log)
                    if early_stopping.early_stop:
                        logging.info("Early stopping")
                        break
                    epoch_log.write(
                        "----------------------------------------------------------------------------------------\n")
                    epoch_log.flush()
                    learning_rate_adapter(model_optim, vali_loss)

                # rename the best_vali to final_best_vali
                os.rename(cv_path + '/' + 'best_vali.pth', cv_path + '/' + 'final_best_vali.pth')

                logging.info("Loading the best validation model!")
                self.model.load_state_dict(torch.load(cv_path + '/' + 'final_best_vali.pth'))
                # model.eval()
                test_time = time.time()
                test_loss, test_acc, test_f_w, test_f_macro, test_f_micro = self.validation(self.model, test_loader,
                                                                                            criterion, iter + 1)
                logging.info("Test for CV: {} cost time: {}".format(iter, time.time() - test_time))
                logging.info(
                    "Final Test Performance : Test Accuracy: {0:.7f}  Test weighted F1: {1:.7f}  Test macro F1 {2:.7f} ".format(
                        test_acc, test_f_w, test_f_macro))
                epoch_log.write(
                    "Final Test Performance : Test weighted F1: {0:.7f}  Test macro F1 {1:.7f}\n\n\n\n\n\n\n\n".format(
                        test_f_w, test_f_macro))
                epoch_log.flush()

                score_log.write("Test weighted F1: {0:.7f}  Test macro F1 {1:.7f}\n".format(test_f_w, test_f_macro))
                score_log.flush()

                epoch_log.close()
                score_log.close()

            # ------------------------------ code for  regularization and fine tuning -----------------------------------------------------------------

            if self.args.wavelet_filtering_finetuning:
                finetuned_score_log_file_name = os.path.join(self.path, "finetuned_score.txt")
                if skip_finetuning:
                    logging.info("================Skip the {} CV Experiment Fine Tuning================".format(iter))
                else:
                    # thre_index : selected number
                    epoch_log = open(epoch_log_file_name, "a")
                    epoch_log.write(
                        "----------------------------------------------------------------------------------------\n")
                    epoch_log.write(
                        "--------------------------------------Fine Tuning-----------------------------------------\n")
                    epoch_log.write(
                        "----------------------------------------------------------------------------------------\n")

                    self.model = self.build_model().to(self.device)
                    self.model.load_state_dict(torch.load(cv_path + '/' + 'final_best_vali.pth'))

                    finetuned_score_log = open(finetuned_score_log_file_name, "a")

                    thre_index = int(self.args.f_in * self.args.wavelet_filtering_finetuning_percent) - 1
                    gamma_weight = self.model.gamma.squeeze().abs().clone()
                    sorted_gamma_weight, i = torch.sort(gamma_weight, descending=True)
                    threshold = sorted_gamma_weight[thre_index]
                    mask = gamma_weight.data.gt(threshold).float().to(self.device)
                    idx0 = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    # build the new model
                    new_model = model_builder(self.args, input_f_channel=thre_index).to(self.device)

                    logging.info("------------Fine Tuning  : ", self.args.f_in - thre_index,
                                 "  will be pruned   -----------------------------------------")
                    logging.info("old model Parameter : {}".format(self.model_size))
                    logging.info("pruned model Parameter : {}".format(
                        np.sum([para.numel() for para in new_model.parameters()])))
                    logging.info(
                        "----------------------------------------------------------------------------------------")
                    # copy the weights
                    flag_channel_selection = False
                    for n, p in new_model.named_parameters():
                        if "wavelet_conv" in n:
                            p.data = self.model.state_dict()[n].data[idx0.tolist(), :, :, :].clone()
                        elif n == "gamma":
                            flag_channel_selection = True
                            p.data = self.model.state_dict()[n].data[:, idx0.tolist(), :, :].clone()
                        elif flag_channel_selection and "conv" in n:
                            p.data = self.model.state_dict()[n].data[:, idx0.tolist(), :, :].clone()
                            flag_channel_selection = False
                        else:
                            p.data = self.model.state_dict()[n].data.clone()

                    early_stopping = EarlyStopping(patience=15, verbose=True)
                    learning_rate_adapter = adjust_learning_rate_class(self.args, True)
                    model_optim = optim.Adam(new_model.parameters(), lr=0.0001)
                    criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
                    for epoch in range(self.args.train_epochs):
                        train_loss = []
                        new_model.train()
                        epoch_time = time.time()

                        for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):
                            batch_x1 = batch_x1.double().to(self.device)

                            batch_y = batch_y.long().to(self.device)
                            outputs = new_model(batch_x1)

                            loss = criterion(outputs, batch_y)

                            train_loss.append(loss.item())

                            model_optim.zero_grad()
                            loss.backward()
                            model_optim.step()

                        logging.info("Fine Tuning Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                        epoch_log.write(
                            "Fine Tuning Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                        epoch_log.write("\n")

                        train_loss = np.average(train_loss)
                        vali_loss, vali_acc, vali_f_w, vali_f_macro, vali_f_micro = self.validation(new_model,
                                                                                                    val_loader,
                                                                                                    criterion)

                        logging.info(
                            "Fine Tuning VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} ".format(
                                epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))

                        epoch_log.write(
                            "Fine Tuning VALI: Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Vali Accuracy: {4:.7f}  Vali weighted F1: {5:.7f}  Vali macro F1 {6:.7f} \n".format(
                                epoch + 1, train_steps, train_loss, vali_loss, vali_acc, vali_f_w, vali_f_macro))

                        early_stopping(vali_loss, new_model, cv_path, vali_f_macro, vali_f_w, epoch_log)
                        if early_stopping.early_stop:
                            logging.info("Early stopping")
                            break
                        epoch_log.write(
                            "----------------------------------------------------------------------------------------\n")
                        epoch_log.flush()
                        learning_rate_adapter(model_optim, vali_loss)
                    # rename the best_vali to final_best_vali
                    os.rename(cv_path + '/' + 'best_vali.pth', cv_path + '/' + 'final_finetuned_best_vali.pth')

                    logging.info("Loading the best finetuned validation model!")
                    new_model.load_state_dict(torch.load(cv_path + '/' + 'final_finetuned_best_vali.pth'))

                    test_loss, test_acc, test_f_w, test_f_macro, test_f_micro = self.validation(new_model, test_loader,
                                                                                                criterion)
                    logging.info(
                        "Fine Tuning Final Test Performance : Test Accuracy: {0:.7f}  Test weighted F1: {1:.7f}  Test macro F1 {2:.7f} ".format(
                            test_acc, test_f_w, test_f_macro))
                    epoch_log.write(
                        "Final Test Performance : Test weighted F1: {0:.7f}  Test macro F1 {1:.7f}\n\n\n\n\n\n\n\n".format(
                            test_f_w, test_f_macro))
                    epoch_log.flush()

                    finetuned_score_log.write(
                        "Test weighted F1: {0:.7f}  Test macro F1 {1:.7f}\n".format(test_f_w, test_f_macro))
                    finetuned_score_log.flush()

                    epoch_log.close()
                    finetuned_score_log.close()

    def prediction_test(self):

        # assert self.args.exp_mode == "Given"
        # load the data
        dataset = data_dict[self.args.test_data_name](self.args)
        num_of_cv = dataset.num_of_cv

        for iter in range(num_of_cv):
            self.model = self.build_model().to(self.device)
            setting = self.get_setting_name()
            path = os.path.join(self.args.to_save_path, 'logs/' + setting)
            self.model.load_state_dict(torch.load(os.path.join(path, 'cv_' + str(iter) + '/final_best_vali.pth')))
            dataset.update_train_val_test_keys()
            test_loader = self._get_data(dataset, flag='test', weighted_sampler=self.args.weighted_sampler)
            criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
            self.model.eval()
            total_loss = []
            preds = []
            trues = []
            logging.info("================ {} CV ======================".format(iter))
            # Check if csv file already esists, if it does then skip prediction
            if os.path.exists(os.path.join(path, 'cv_' + str(iter) + '/prediction_result_' +
                                                 self.args.device + '_' + self.args.test_device + '.csv')):
                print("Prediction already done for this setting, skipping...")
                continue
            with torch.no_grad():
                for i, (batch_x1, batch_x2, batch_y) in enumerate(test_loader):
                    ''' If the test data and the model do not have the same frequency, we will have to resample using interpolation'''
                    if self.args.test_windowsize != self.args.windowsize:
                        target_size = self.args.windowsize

                        input_tensor_reshaped = batch_x1.view(batch_x1.shape[0], self.args.test_windowsize,
                                                              self.args.c_in)
                        input_tensor_reshaped = torch.tensor(input_tensor_reshaped, dtype=torch.float32)

                        interpolated_tensor = F.interpolate(input_tensor_reshaped.permute(0, 2, 1), size=target_size,
                                                            mode='linear', align_corners=False).permute(0, 2, 1)

                        batch_x1 = interpolated_tensor.view(batch_x1.shape[0], 1, self.args.windowsize,
                                                            self.args.c_in)

                    if "cross" in self.args.model_type:
                        batch_x1 = batch_x1.double().to(self.device)
                        batch_x2 = batch_x2.double().to(self.device)
                        batch_y = batch_y.long().to(self.device)
                        # model prediction
                        if self.args.output_attention:
                            outputs = self.model(batch_x1, batch_x2)[0]
                        else:
                            outputs = self.model(batch_x1, batch_x2)
                    else:
                        batch_x1 = batch_x1.double().to(self.device)
                        batch_y = batch_y.long().to(self.device)

                        # model prediction
                        if self.args.output_attention:
                            outputs = self.model(batch_x1)[0]
                        else:
                            outputs = self.model(batch_x1)

                    pred = outputs.detach()  # .cpu()
                    true = batch_y.detach()  # .cpu()

                    loss = criterion(pred, true)
                    total_loss.append(loss.cpu())

                    preds.extend(list(np.argmax(outputs.detach().cpu().numpy(), axis=1)))
                    trues.extend(list(batch_y.detach().cpu().numpy()))

            acc = accuracy_score(preds, trues)
            f_w = f1_score(trues, preds, average='weighted')
            f_macro = f1_score(trues, preds, average='macro')
            f_micro = f1_score(trues, preds, average='micro')
            logging.info(
                "Test Accuracy: {0:.7f}  Test weight F1: {1:.7f}  Test macro F1 {2:.7f}  Test micro F1 {3:.7f}".format(
                    acc, f_w, f_macro, f_micro))
            prediction_result_df = pd.DataFrame({'preds': preds, 'trues': trues})
            prediction_result_df.to_csv(os.path.join(path, 'cv_' + str(iter) + '/prediction_result_' +
                                                     self.args.device + '_' + self.args.test_device + '.csv'))

        return preds, trues

    def validation(self, model, data_loader, criterion, index_of_cv=None, selected_index=None):
        model.eval()
        total_loss = []
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x1, batch_x2, batch_y) in enumerate(data_loader):

                if "cross" in self.args.model_type:
                    batch_x1 = batch_x1.double().to(self.device)
                    batch_x2 = batch_x2.double().to(self.device)
                    batch_y = batch_y.long().to(self.device)
                    # model prediction
                    if self.args.output_attention:
                        outputs = model(batch_x1, batch_x2)[0]
                    else:
                        outputs = model(batch_x1, batch_x2)
                else:
                    if selected_index is None:
                        batch_x1 = batch_x1.double().to(self.device)
                    else:
                        batch_x1 = batch_x1[:, selected_index.tolist(), :, :].double().to(self.device)
                    batch_y = batch_y.long().to(self.device)

                    # model prediction
                    if self.args.output_attention:
                        outputs = model(batch_x1)[0]
                    else:
                        outputs = model(batch_x1)

                pred = outputs.detach()  # .cpu()
                true = batch_y.detach()  # .cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.cpu())

                preds.extend(list(np.argmax(outputs.detach().cpu().numpy(), axis=1)))
                trues.extend(list(batch_y.detach().cpu().numpy()))

        total_loss = np.average(total_loss)
        acc = accuracy_score(preds, trues)
        # f_1 = f1_score(trues, preds)
        f_w = f1_score(trues, preds, average='weighted')
        f_macro = f1_score(trues, preds, average='macro')
        f_micro = f1_score(trues, preds, average='micro')
        if index_of_cv:
            cf_matrix = confusion_matrix(trues, preds, labels=[i for i in range(self.args.num_classes)], )
            cf_matrix = pd.DataFrame(cf_matrix, columns=[i for i in range(self.args.num_classes)],
                                     index=[i for i in range(self.args.num_classes)])
            # with open("{}.npy".format(index_of_cv), 'wb') as f:
            #    np.save(f, cf_matrix)
            plt.figure()
            sns.heatmap(cf_matrix, annot=True, fmt='d')
            plt.xlabel('True Labels')
            plt.ylabel('Predicted Labels')
            # plt.savefig("{}.png".format(index_of_cv))
        model.train()

        return total_loss, acc, f_w, f_macro, f_micro  # , f_1
