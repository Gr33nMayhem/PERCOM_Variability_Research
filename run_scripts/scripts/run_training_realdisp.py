import sys
import os
import argparse

sys.path.append(os.path.join("..", ".."))
from run_scripts.scripts.train_with_data_realdisp import run_train_process_with_data

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, help='Device Name for training')
parser.add_argument('--variant', type=str, help='Device Name for training')
parser.add_argument('--freq', type=str, help='New sampling freq if freq needs to be modified')
parser.add_argument('--noise', type=str, help='Remove noise from data, Y/N')
parser.add_argument('--norm', type=str, help='Normalize type standardization/minmax')

args = parser.parse_args()

run_train_process_with_data(args.device, args.variant, args.freq, args.noise, args.norm)
