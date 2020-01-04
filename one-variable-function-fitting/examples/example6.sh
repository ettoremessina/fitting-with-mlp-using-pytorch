#!/bin/bash

FX="np.arctan(x)"
RB=-5.0
RE=5.0

python ../fx_gen.py --dsout datasets/example6_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example6_train.csv --modelout models/example6.pth \
  --hlayers 100 150 --hactivation 'Tanh()' 'ReLU()' \
  --epochs 500 --batch_size 100 \
  --optimizer 'SGD(momentum=0.0)' --learning_rate=1e-2


python ../fx_gen.py --dsout datasets/example6_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example6.pth --testds datasets/example6_test.csv --predicted predictions/example6_pred.csv

python ../fx_plot.py --trainds datasets/example6_train.csv --predicted predictions/example6_pred.csv
