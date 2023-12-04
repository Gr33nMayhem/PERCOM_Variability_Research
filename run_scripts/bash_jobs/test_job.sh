#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0
#SBATCH --mail-user=Gr33nMayhem@gmail.com
#SBATCH --mail-type=ALL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done
cd ~/$projects/def-plago/akhaked/PERCOM_Variability_Model_Research
module purge
source ~/py310/bin/activate
python run_scripts/test_scripts/test_run.py
