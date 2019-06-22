#!/usr/bin/env bash

# replace with your anaconda env path
source /home/pollo/miniconda3/bin/activate nn

python glove_keras/train.py data/text8 -o output/vectors_text8_2_300.txt --epochs 15 --size 300 --min-count 5 --max-vocab 100000 --window 2 --x-max 100 --batch-size 16384

python glove_keras/train.py data/text8 -o output/vectors_text8_4_300.txt --epochs 15 --size 300 --min-count 5 --max-vocab 100000 --window 4 --x-max 100 --batch-size 16384

python glove_keras/train.py data/text8 -o output/vectors_text8_6_300.txt --epochs 15 --size 300 --min-count 5 --max-vocab 100000 --window 6 --x-max 100 --batch-size 16384

python glove_keras/train.py data/text8 -o output/vectors_text8_10_300.txt --epochs 15 --size 300 --min-count 5 --max-vocab 100000 --window 10 --x-max 100 --batch-size 16384
