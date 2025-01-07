#!/bin/bash
#SBATCH --job-name=eval_gta  # create a short name for your job
#SBATCH --partition=HGX             # specify the partition name: gpu
#SBATCH --qos=lv1
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G               # total memory (RAM) per node
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --output=output/out-%j.out      # output format
#SBATCH --error=output/error-out-%j.out      # error output file
#SBATCH --account=engineering
#SBATCH --dependency=7787293
#--------------------task  part-------------------------


## clean env
module purge
## load environment need by this task
module load slurm/BigAI/23.02.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhangbofei/anaconda3/lib
source /home/zhangbofei/anaconda3/bin/activate  # commented out by conda initialize
conda init
conda activate agent_tune
export AGENT_CONFIG='configs/agent_config.yaml' 
python examples/gta/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7879870_2024_09_30_15_55/ --disable-vision
# python examples/gta/main.py --engine tonggpt --disable-vision