#!/bin/bash

#Archimedean spiral on the space

FXT="0.1 * t * np.cos(t)"
FYT="0.1 * t * np.sin(t)"
FZT="t"

python ../pmc3t_gen.py --dsout datasets/example1_train.csv --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc3t_fit.py --trainds datasets/example1_train.csv --modelout models/example1.pth \
  --hlayers 200 300 200 --hactivation 'Sigmoid()' 'Tanh()' 'Sigmoid()' \
  --epochs 250

python ../pmc3t_gen.py --dsout datasets/example1_test.csv  --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc3t_predict.py --model models/example1.pth --ds datasets/example1_test.csv --predictionout predictions/example1_pred.csv

python ../pmc3t_plot.py --ds datasets/example1_test.csv --prediction predictions/example1_pred.csv
#python ../pmc3t_plot.py --ds datasets/example1_train.csv --prediction predictions/example1_pred.csv --savefig predictions/example1.png
