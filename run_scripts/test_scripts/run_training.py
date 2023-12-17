import sys
import os
import argparse

sys.path.append(os.path.join("..", ".."))
from run_scripts.test_scripts.train_with_data import run_train_process_with_data

parser = argparse.ArgumentParser(
    description='Enter a sensor name as --device <device name> \n ' +
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
parser.add_argument('--device', type=str, help='Device Name for training')

args = parser.parse_args()

run_train_process_with_data(args.device)
