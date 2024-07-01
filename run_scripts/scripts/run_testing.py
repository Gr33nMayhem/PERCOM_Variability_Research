import sys
import os
import argparse

sys.path.append(os.path.join("..", ".."))
from run_scripts.scripts.test_with_data import run_test_process_with_data

parser = argparse.ArgumentParser()
parser.add_argument('--device1', type=str, help='Device 1 Name for training')
parser.add_argument('--device2', type=str, help='Device 2 Name for training')
parser.add_argument('--freq', type=str, help='New sampling freq if freq needs to be modified')
parser.add_argument('--noise', type=str, help='Remove noise from data, Y/N')
parser.add_argument('--norm', type=str, help='Normalize type standardization/minmax')

device1 = parser.parse_args().device1
device2 = parser.parse_args().device2
freq = parser.parse_args().freq
noise = parser.parse_args().noise
norm = parser.parse_args().norm

if device1 is not None and device2 is not None and device1 != "" and device2 != "":
    run_test_process_with_data(device1, device1, freq, noise, norm)
    run_test_process_with_data(device2, device2, freq, noise, norm)
    run_test_process_with_data(device1, device2, freq, noise, norm)
    run_test_process_with_data(device2, device1, freq, noise, norm)


else:
    devices = ['empatica-left', 'empatica-right', 'bluesense-LUA', 'bluesense-LWR', 'bluesense-RUA', 'bluesense-RWR1',
               'bluesense-RWR2', 'bluesense-TRS', 'maxim-red', 'maxim-green']

    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            run_test_process_with_data(devices[i], devices[j], freq, noise, norm)
            run_test_process_with_data(devices[j], devices[i], freq, noise, norm)

    for i in range(len(devices)):
        run_test_process_with_data(devices[i], devices[i])
