#!/bin/bash

#Bernoulli's spiral

FXT="np.exp(0.1 * t) * np.cos(t)"
FYT="np.exp(0.1 * t) * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example2_2_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc2t_fit_2.py --trainds datasets/example2_2_train.csv --modelout models/example2_2 \
  --hlayers 200 200 200 --hactivation sigmoid sigmoid sigmoid \
  --epochs 500 \
  --optimizer 'Adamax()'

python ../pmc2t_gen.py --dsout datasets/example2_2_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc2t_predict_2.py --model models/example2_2 --testds datasets/example2_2_test.csv --predicted predictions/example2_2_pred.csv

python ../pmc2t_plot.py --ds datasets/example2_2_test.csv --predicted predictions/example2_2_pred.csv
#python ../pmc2t_plot.py --ds datasets/example2_2_train.csv --predicted predictions/example2_2_pred.csv --savefig predictions/example2_2.png
