import sys
import os

# import argparse

sys.path.append(os.path.join("..", ".."))
from run_scripts.scripts.test_with_data_realdisp import run_test_process_with_data_realdisp

# parser = argparse.ArgumentParser()
# parser.add_argument('--device1', type=str, help='Device 1 Name for training')
# parser.add_argument('--device2', type=str, help='Device 2 Name for training')

device = "RLA"
variant1 = "ideal"
variant2 = "self"

run_test_process_with_data_realdisp(device, variant1, variant1)
run_test_process_with_data_realdisp(device, variant2, variant2)
run_test_process_with_data_realdisp(device, variant1, variant2)
run_test_process_with_data_realdisp(device, variant2, variant1)
