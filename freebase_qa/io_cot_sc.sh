#!/bin/bash
set -e  # 一条命令失败就退出

python -u main.py --dataset webqsp --method io
python -u main.py --dataset cwq --method io
python -u main.py --dataset grailqa --method io
python -u main.py --dataset webq --method io

python -u main.py --dataset webqsp --method cot
python -u main.py --dataset cwq --method cot
python -u main.py --dataset grailqa --method cot
python -u main.py --dataset webq --method cot
