import sys
import os
import argparse

sys.path.append(os.path.join("..", ".."))
from run_scripts.scripts.test_with_data import run_test_process_with_data

parser = argparse.ArgumentParser()
parser.add_argument('--device1', type=str, help='Device 1 Name for training')
parser.add_argument('--device2', type=str, help='Device 2 Name for training')

device1 = parser.parse_args().device1
device2 = parser.parse_args().device2

if device1 is not None and device2 is not None:
    devices = ['empatica-left', 'empatica-right', 'bluesense-LUA', 'bluesense-LWR', 'bluesense-RUA', 'bluesense-RWR1',
               'bluesense-RWR2', 'bluesense-TRS', 'maxim-red', 'maxim-green']

    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            run_test_process_with_data(devices[i], devices[j])
            run_test_process_with_data(devices[j], devices[i])

    for i in range(len(devices)):
        run_test_process_with_data(devices[i], devices[i])

else:
    run_test_process_with_data(device1, device2)
    run_test_process_with_data(device2, device1)
