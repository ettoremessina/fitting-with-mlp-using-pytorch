#!/bin/bash

FXT="np.cos(t)"
FYT="np.sin(t)"
FZT="np.cos(4 * t) / 4.0"

python ../pmc3t_gen.py --dsout datasets/example3_train.csv --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 6.28 --rstep 0.01
python ../pmc3t_fit.py --trainds datasets/example3_train.csv --modelout models/example3.pth \
--hlayers 150 300 150 --hactivation 'Tanh()' 'Tanh()' 'Tanh()' \
--epochs 500 \
--optimizer 'Adamax(lr=0.005)'

python ../pmc3t_gen.py --dsout datasets/example3_test.csv  --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 6.28 --rstep 0.0275
python ../pmc3t_predict.py --model models/example3.pth --ds datasets/example3_test.csv --predictionout predictions/example3_pred.csv

python ../pmc3t_plot.py --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv
#python ../pmc3t_plot.py --ds datasets/example3_train.csv --prediction predictions/example3_pred.csv --savefig predictions/example3.png
