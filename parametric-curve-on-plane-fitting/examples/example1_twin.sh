#!/bin/bash

#Archimedean spiral

FXT="0.1 * t * np.cos(t)"
FYT="0.1 * t * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example1_twin_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc2t_fit_twin.py --trainds datasets/example1_twin_train.csv --modelout models/example1_twin.pth \
  --hlayers 200 300 200 --hactivation 'Sigmoid()' 'Tanh()' 'Sigmoid()' \
  --epochs 250

python ../pmc2t_gen.py --dsout datasets/example1_twin_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc2t_predict_twin.py --model models/example1_twin.pth --ds datasets/example1_twin_test.csv --predicted predictions/example1_twin_pred.csv

python ../pmc2t_plot.py --ds datasets/example1_twin_test.csv --predicted predictions/example1_twin_pred.csv
#python ../pmc2t_plot.py --ds datasets/example1_twin_train.csv --predicted predictions/example1_pred_twin.csv --savefig predictions/example1_twin.png
