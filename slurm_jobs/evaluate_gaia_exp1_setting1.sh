#!/bin/bash
#SBATCH --job-name=eval_gaia  # create a short name for your job
#SBATCH --partition=DGX,HGX             # specify the partition name: gpu
#SBATCH --qos=lv0a
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G               # total memory (RAM) per node
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --output=output/out-%j.out      # output format
#SBATCH --error=output/error-out-%j.out      # error output file
#SBATCH --account=engineering
#SBATCH --dependency=7880273
#--------------------task  part-------------------------


## clean env
module purge
## load environment need by this task
module load slurm/BigAI/23.02.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhangbofei/anaconda3/lib
source /home/zhangbofei/anaconda3/bin/activate  # commented out by conda initialize
conda init
conda activate agent_tune
export RUN_MODE=eval
python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7897481_2024_11_10_07_40/ --data-name 2023_level1 --split validation


# CUDA_VISIBLE_DEVICES=1 python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7897481_2024_11_10_07_40/ --data-name 2023_level3 --split validation > eval_settings1_lv3.log 2>&1 &
#python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7880273_2024_10_01_16_09/ --data-name 2023_level3 --split validation

#python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7880273_2024_10_01_16_09/ --data-name 2023_level2 --split validation


# python examples/gaia/main.py --engine minicpm --data-name 2023_level1 --split validation

# python examples/gaia/main.py --engine tonggpt --data-name 2023_level1 --split validation

#python examples/gaia/main.py --engine tonggpt --data-name 2023_level3 --split validation
