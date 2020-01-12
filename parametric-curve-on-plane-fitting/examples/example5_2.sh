#!/bin/bash

#Lissajous

FXT="2 * np.sin(0.5 * t + 1)"
FYT="3 * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example5_2_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.01
python ../pmc2t_fit_2.py --trainds datasets/example5_2_train.csv --modelout models/example5_2 \
  --hlayers 100 100 --hactivation relu relu \
  --epochs 500

python ../pmc2t_gen.py --dsout datasets/example5_2_test.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.0475
python ../pmc2t_predict_2.py --model models/example5_2 --testds datasets/example5_2_test.csv --predicted predictions/example5_2_pred.csv

python ../pmc2t_plot.py --ds datasets/example5_2_test.csv --predicted predictions/example5_2_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example5_2_test.csv --predicted predictions/example5_2_pred.csv --savefig predictions/example5_2.png
