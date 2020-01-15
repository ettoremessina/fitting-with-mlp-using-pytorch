#!/bin/bash

#Bernoulli's spiral

FXT="np.exp(0.1 * t) * np.cos(t)"
FYT="np.exp(0.1 * t) * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example2_1_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc2t_fit_1.py --trainds datasets/example2_1_train.csv --modelout models/example2_1.pth \
  --hlayers 200 200 200 --hactivation 'Sigmoid()' 'Sigmoid()' 'Sigmoid()' \
  --epochs 500 \
  --optimizer 'Adamax()'

python ../pmc2t_gen.py --dsout datasets/example2_1_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc2t_predict_1.py --model models/example2_1.pth --testds datasets/example2_1_test.csv --predicted predictions/example2_1_pred.csv

python ../pmc2t_plot.py --ds datasets/example2_1_test.csv --predicted predictions/example2_1_pred.csv
#python ../pmc2t_plot.py --ds datasets/example2_1_train.csv --predicted predictions/example2_1_pred.csv --savefig predictions/example2_1.png
