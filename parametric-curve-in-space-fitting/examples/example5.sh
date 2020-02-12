#!/bin/bash

FXT="np.exp(2.0 - t)"
FYT="np.sqrt(t)"
FZT="np.log(1.0 + t)"

python ../pmc3t_gen.py --dsout datasets/example5_train.csv --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 10.0 --rstep 0.01
python ../pmc3t_fit.py --trainds datasets/example5_train.csv --modelout models/example5.pth \
  --hlayers 100 100 100 --hactivation 'ReLU()' 'ReLU()' 'ReLU()' \
  --epochs 500 --batch_size 50 \
  --optimizer 'Adamax(lr=0.05)'

python ../pmc3t_gen.py --dsout datasets/example5_test.csv --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 10.0 --rstep 0.0475
python ../pmc3t_predict.py --model models/example5.pth --ds datasets/example5_test.csv --predictionout predictions/example5_pred.csv

python ../pmc3t_plot.py --ds datasets/example5_test.csv --prediction predictions/example5_pred.csv
#python ../pmc2t_plot.py --ds datasets/example5_test.csv --prediction predictions/example5_pred.csv --savefig predictions/example5.png
