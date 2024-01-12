import sys
import os
import argparse

sys.path.append(os.path.join("..", ".."))
from run_scripts.test_scripts.test_with_data import run_test_process_with_data

devices = ['empatica-left', 'empatica-right', 'bluesense-LUA', 'bluesense-LWR', 'bluesense-RUA', 'bluesense-RWR1',
           'bluesense-RWR2', 'bluesense-TRS', 'maxim-red', 'maxim-green']

for i in range(len(devices)):
    for j in range(i + 1, len(devices)):
        run_test_process_with_data(devices[i], devices[j])
        run_test_process_with_data(devices[j], devices[i])
