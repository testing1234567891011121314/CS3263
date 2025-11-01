#!/bin/bash

## Change this to a job name you want
#SBATCH --job-name=inference

# --- Resource requests ---
#SBATCH --partition=gpu            # use the GPU partition
#SBATCH --ntasks=1                 # one task
#SBATCH --cpus-per-task=8          # 24 CPU core
#SBATCH --mem=24G                  # memory per node
#SBATCH --time=2:59:00             # runtime limit is 3 hours

##SBATCH --gres=gpu:a100-40:1       # request one A100 GPU
##SBATCH --constraint=xgph          # ensure A100 host

#SBATCH --gres=gpu:h100-96:1       # request one H100 GPU
#SBATCH --constraint=xgpi          # ensure H100 host

## Just useful logfile names
#SBATCH --output=inference_%j.slurmlog
#SBATCH --error=inference_%j.slurmlog

# Print debug info
echo "Running on host: $(hostname)"
echo "Time: $(date)"
echo "Working directory: $(pwd)"

# Change to the directory where inference.py lives
cd $PWD/WLASL/code/I3D
echo "Changed working directory to $(pwd)"

# Define the venv directory
VENV_DIR=/home/y/yaohejun/cs3263/project/CS3263/CS3263venv
# VENV_DIR=$PWD/CS3263venv
# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

# Run your Python program here
python3 inference.py 69206.mp4

# Deactivate venv (optional)
deactivate
echo "Inference complete."