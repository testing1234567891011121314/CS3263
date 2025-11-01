#!/bin/bash
#SBATCH --job-name=make_venv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=make_venv.out
#SBATCH --error=make_venv.err

# Print useful debug info
echo "Running on host: $(hostname)"
echo "Time: $(date)"
echo "Current working directory: $(pwd)"

# Define the venv directory
VENV_DIR=$PWD/CS3263venv

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip and install packages (optional)
# pip install --upgrade pip setuptools wheel

# Example: install requirements if you have a requirements.txt
if [ -f requirements.txt ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# (Optional) Run your Python program here
# python your_script.py

echo "Virtual environment setup complete."
deactivate
