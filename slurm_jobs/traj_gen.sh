#!/bin/bash
#SBATCH --job-name=chunk_16  # create a short name for your job
#SBATCH --partition=DGX             # specify the partition name: gpu
#SBATCH --qos=lv2
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G               # total memory (RAM) per node
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=output/out-%j.out      # output format
#SBATCH --error=output/error-out-%j.out      # error output file
#SBATCH --account=engineering
#--------------------task  part-------------------------


## clean env
module purge
## load environment need by this task
module load slurm/BigAI/23.02.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhangbofei/anaconda3/lib
# source /home/zhangbofei/anaconda3/bin/activate  # commented out by conda initialize

conda activate agent_tune
python data_generation/chunk_traj_generation/gta4_traj_genetation_chunk.py --chunk 16

