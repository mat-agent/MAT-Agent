#!/bin/bash
#SBATCH --job-name=qwen2.5_72b  # create a short name for your job
#SBATCH --partition=DGX             # specify the partition name: gpu
#SBATCH --qos=lv2
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G               # total memory (RAM) per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --output=output/out-%j.out      # output format
#SBATCH --error=output/error-out-%j.out      # error output file
#SBATCH --account=engineering
#--------------------task  part-------------------------


## clean env
module purge
## load environment need by this task
module load slurm/BigAI/23.02.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhangbofei/anaconda3/lib
source /home/zhangbofei/anaconda3/bin/activate  # commented out by conda initialize
conda init
conda activate agent_tune
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size $GPUS_PER_NODE --dtype bfloat16 --gpu-memory-utilization 0.98 --max-model-len 20000

# python tests/test_vllm.py