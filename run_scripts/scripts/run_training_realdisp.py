import sys
import os
import argparse

sys.path.append(os.path.join("..", ".."))
from run_scripts.scripts.train_with_data_realdisp import run_train_process_with_data

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, help='Device Name for training')
parser.add_argument('--variant', type=str, help='Device Name for training')

args = parser.parse_args()

run_train_process_with_data(args.device, args.variant)
