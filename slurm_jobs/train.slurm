#!/bin/bash
#SBATCH -J gaia
#SBATCH -p HGX
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -q lv0b
#SBATCH --time=10:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

module load anaconda3/2021.11
source activate tongagent

python -m data_generation.gaia_pipeline.gaia21_file_content2file_openai >& gaia_3kfile.log
