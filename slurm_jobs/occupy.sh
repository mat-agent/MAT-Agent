srun --time=72:00:00 --partition=HGX,DGX --qos=lv1 --mem=64G --account=engineering --gres=gpu:8 --cpus-per-task=32 --pty bash  -c '
    echo "Starting interactive session..."
    module load slurm/BigAI/23.02.2
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhangbofei/anaconda3/lib
    source /home/zhangbofei/anaconda3/bin/activate  # commented out by conda initialize
    conda activate agent_tune
    # Execute any other commands you need
    echo "Environment ready!"
    cd /scratch/zhangbofei/Projects/Multimodal-CL/iclr_09/TongAgent
    GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
    nvidia-smi
    python scripts/report.py
    vllm serve /scratch/ml/zhangxintong/A_Models/Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size $GPUS_PER_NODE --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 20000 &
    # Start an interactive shell
    exec bash -i
'
