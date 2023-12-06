#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --gpus-per-node=v100:4
#SBATCH --mem=32G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=15:0:0
#SBATCH --mail-user=Gr33nMayhem@gmail.com
#SBATCH --mail-type=ALL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done
cd ~/projects/def-plago/akhaked/PERCOM_Variability_Model_Research/run_scripts/test_scripts
module purge
source ~/py310/bin/activate
python run_training.py
