#!/bin/bash

#Lissajous

FXT="2 * np.sin(0.5 * t + 1)"
FYT="3 * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example5_1_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.01
python ../pmc2t_fit_1.py --trainds datasets/example5_1_train.csv --modelout models/example5_1 \
  --hlayers 100 100 --hactivation relu relu \
  --epochs 500

python ../pmc2t_gen.py --dsout datasets/example5_1_test.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.0475
python ../pmc2t_predict_1.py --model models/example5_1 --testds datasets/example5_1_test.csv --predicted predictions/example5_1_pred.csv

python ../pmc2t_plot.py --ds datasets/example5_1_test.csv --predicted predictions/example5_1_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example5_1_test.csv --predicted predictions/example5_1_pred.csv --savefig predictions/example5_1.png
