#!/bin/bash

#Archimedean spiral

FXT="0.1 * t * np.cos(t)"
FYT="0.1 * t * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example1_2_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc2t_fit_2.py --trainds datasets/example1_2_train.csv --modelout models/example1_2.pth \
  --hlayers 200 300 200 --hactivation 'Sigmoid()' 'Tanh()' 'Sigmoid()' \
  --epochs 1000

python ../pmc2t_gen.py --dsout datasets/example1_2_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc2t_predict_2.py --model models/example1_2.pth --testds datasets/example1_2_test.csv --predicted predictions/example1_2_pred.csv

python ../pmc2t_plot.py --ds datasets/example1_2_test.csv --predicted predictions/example1_2_pred.csv
#python ../pmc2t_plot.py --ds datasets/example1_2_train.csv --predicted predictions/example1_pred_2.csv --savefig predictions/example1_2.png
