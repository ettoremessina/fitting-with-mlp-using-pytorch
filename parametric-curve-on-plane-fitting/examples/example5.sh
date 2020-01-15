#!/bin/bash

#Lissajous

FXT="2 * np.sin(0.5 * t + 1)"
FYT="3 * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example5_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.01
python ../pmc2t_fit.py --trainds datasets/example5_train.csv --modelout models/example5.pth \
  --hlayers 150 150 --hactivation 'ReLU()' 'ReLU()' \
  --epochs 400 \
  --optimizer 'Adamax(lr=0.01)'

python ../pmc2t_gen.py --dsout datasets/example5_test.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.0475
python ../pmc2t_predict.py --model models/example5.pth --ds datasets/example5_test.csv --predicted predictions/example5_pred.csv

python ../pmc2t_plot.py --ds datasets/example5_test.csv --predicted predictions/example5_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example5_test.csv --predicted predictions/example5_pred.csv --savefig predictions/example5.png
