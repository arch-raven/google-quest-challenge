#!/bin/bash
#PBS -l select=ncpus=8:mem=16gb:ngpus=1
#PBS -q gpu
module load cuda
module load anaconda/3
source activate torchenv
cd google-quest-challenge/
python src/main.py --fold 0 --model_name ccluster
python src/main.py --fold 1 --model_name ccluster
python src/main.py --fold 2 --model_name ccluster
python src/main.py --fold 3 --model_name ccluster
python src/main.py --fold 4 --model_name ccluster
source deactivate
