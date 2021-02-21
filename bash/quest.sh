#!/bin/bash
#PBS -l select=ncpus=8:mem=16gb:ngpus=1
#PBS -q gpu
module load cuda
module load anaconda/3
source activate torchenv
cd google-quest-challenge/
python src/main.py --fold 0
python src/main.py --fold 1
python src/main.py --fold 2
python src/main.py --fold 3
python src/main.py --fold 4
source deactivate
