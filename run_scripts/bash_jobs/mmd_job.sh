#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:0:0
#SBATCH --mail-user=Gr33nMayhem@gmail.com
#SBATCH --mail-type=ALL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done
cd ~/projects/def-plago/akhaked/PERCOM_Variability_Model_Research/run_scripts/scripts
module purge
module load StdEnv/2020
module load python/3.11 scipy-stack
source ~/py311/bin/activate
python run_mmd.py --dataset "$1" --device_train "$2" --device_test "$3" --base_freq "$4"
