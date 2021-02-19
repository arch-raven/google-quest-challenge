#!/bin/bash
#PBS -l select=ncpus=8:mem=16gb:ngpus=1
#PBS -q gpu
module load cuda
module load anaconda/3
source activate torchenv
cd google-quest-challenge/
python src/main.py 0
python src/main.py 1
python src/main.py 2
python src/main.py 3
python src/main.py 4
source deactivate