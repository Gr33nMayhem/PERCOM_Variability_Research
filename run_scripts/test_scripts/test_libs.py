import sys
import os
sys.path.append(os.path.join("..",".."))
from experiment import Exp
import yaml
import os
import torch
from ptflops import get_model_complexity_info
import logging

logging.basicConfig(level=logging.INFO)

# Check if GPU is available
if torch.cuda.is_available():
    # Get the name of the GPU device
    device = torch.cuda.get_device_name()
    logging.info(f"Using GPU: {device}")
else:
    logging.info("Using CPU")
