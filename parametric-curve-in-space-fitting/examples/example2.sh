#!/bin/bash

#Twisted cubic

FXT="t"
FYT="t ** 2"
FZT="t ** 3"

python ../pmc3t_gen.py --dsout datasets/example2_train.csv --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 2.0 --rstep 0.001
python ../pmc3t_fit.py --trainds datasets/example2_train.csv --modelout models/example2.pth \
  --hlayers 200 300 200 --hactivation 'Sigmoid()' 'Tanh()' 'Sigmoid()' \
  --epochs 250

python ../pmc3t_gen.py --dsout datasets/example2_test.csv  --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 2.0 --rstep 0.00475
python ../pmc3t_predict.py --model models/example2.pth --ds datasets/example2_test.csv --predictionout predictions/example2_pred.csv

python ../pmc3t_plot.py --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv
#python ../pmc3t_plot.py --ds datasets/example2_train.csv --prediction predictions/example2_pred.csv --savefig predictions/example2.png
