#!/bin/bash

FX="np.exp(x)"
RB=-5.0
RE=5.0
python ../fx_gen.py --dsout datasets/example3_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example3_train.csv --modelout models/example3.pth \
  --hlayers 200 200 --hactivation 'ReLU()' 'ReLU()' \
  --epochs 500 --batch_size 100

python ../fx_gen.py --dsout datasets/example3_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example3.pth --testds datasets/example3_test.csv --predicted predictions/example3_pred.csv

python ../fx_plot.py --ds datasets/example3_test.csv --predicted predictions/example3_pred.csv
#python ../fx_plot.py --ds datasets/example3_test.csv --predicted predictions/example3_pred.csv --savefig predictions/example3.png
