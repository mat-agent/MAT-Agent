#!/bin/bash
#SBATCH --job-name=gta_minicpm  # create a short name for your job
#SBATCH --partition=DGX             # specify the partition name: gpu
#SBATCH --qos=lv2
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G               # total memory (RAM) per node
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --output=output/out-%j.out      # output format
#SBATCH --error=output/error-out-%j.out      # error output file
#SBATCH --account=engineering
#--------------------task  part-------------------------

## clean env
module purge
## load environment need by this task
module load slurm/BigAI/23.02.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhangbofei/anaconda3/lib
source /home/zhangbofei/anaconda3/bin/activate  
conda activate cpm

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=50238

MODEL="openbmb/MiniCPM-V-2_6" # or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
# Dataset upsample x 5
DATA="data/agent_tune_dataset_cpm_8k_gta_with_verifier.json.json"
LLM_TYPE="qwen2" 
# if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm
#if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE=llama3
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

export WANDB_PROJECT=minicpm
now=$(date '+%Y_%m_%d_%H_%M')

torchrun $DISTRIBUTED_ARGS finetune/finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --tune_vision false \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length 10240 \
    --max_slice_nums 9 \
    --eval_steps 100000 \
    --output_dir output/cpm_v2_6_${SLURM_JOB_ID}_${now} \
    --logging_dir output/cpm_v2_6_log_${SLURM_JOB_ID}_${now} \
    --logging_strategy "steps" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "epoch" \
    --save_steps 100000 \
    --save_total_limit 2 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed scripts/ds_config_zero2.json \
    --report_to wandb \
    --num_train_epochs 5 \
    --image_base_path ./



