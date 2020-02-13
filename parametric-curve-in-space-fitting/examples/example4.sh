#!/bin/bash

FXT="np.sqrt(t)"
FYT="np.sqrt(t)"
FZT="t * np.cos(t)"

python ../pmc3t_gen.py --dsout datasets/example4_train.csv --xt "$FXT" --yt "$FYT" --zt "$FZT" --rbegin 0 --rend 9.42 --rstep 0.005
python ../pmc3t_fit.py --trainds datasets/example4_train.csv --modelout models/example4.pth \
  --hlayers 180 180 --hactivation 'Tanh()' 'Tanh()' \
  --epochs 600 --batch_size 100 \
  --optimizer 'Adamax(lr=0.025)'

python ../pmc3t_gen.py --dsout datasets/example4_test.csv  --xt "$FXT" --yt "$FYT"  --zt "$FZT" --rbegin 0 --rend 9.42 --rstep 0.0275
python ../pmc3t_predict.py --model models/example4.pth --ds datasets/example4_test.csv --predictionout predictions/example4_pred.csv

python ../pmc3t_plot.py --ds datasets/example4_test.csv --prediction predictions/example4_pred.csv
#python ../pmc3t_plot.py --ds datasets/example4_test.csv --prediction predictions/example4_test.csv --savefig predictions/example4.png
