#!/bin/bash
#SBATCH --job-name=git_clone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o git_clone.out

git submodule init
git submodule update --init --recursive