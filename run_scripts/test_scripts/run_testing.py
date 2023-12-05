import sys
import os

sys.path.append(os.path.join("..", ".."))
from run_scripts.test_scripts.test_with_data import run_test_process_with_data

sensors = {'bluesense-RWR1': 0,
           'bluesense-RWR2': 1, }

run_test_process_with_data(sensors['bluesense-RWR1'])
run_test_process_with_data(sensors['bluesense-RWR2'])
