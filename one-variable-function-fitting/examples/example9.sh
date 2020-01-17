#!/bin/bash

FX="np.sin(2 * x) / np.exp(x / 5.0)"
RB=-20.0
RE=20.0

python ../fx_gen.py --dsout datasets/example9_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example9_train.csv --modelout models/example9.pth \
  --hlayers 200 200 200 --hactivations 'Sigmoid()' 'Sigmoid()' 'Sigmoid()' \
  --loss 'MSELoss()' \
  --optimizer 'Adamax(lr=2e-2)' \
  --epochs 500 --batch_size 200  \
  --device cpu

python ../fx_gen.py --dsout datasets/example9_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example9.pth --ds datasets/example9_test.csv --predictedout predictions/example9_pred.csv

python ../fx_plot.py --ds datasets/example9_test.csv --predicted predictions/example9_pred.csv
