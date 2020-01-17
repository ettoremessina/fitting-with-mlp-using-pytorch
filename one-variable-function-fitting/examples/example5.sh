#!/bin/bash

FX="np.log(1+np.abs(x))"
RB=-5.0
RE=5.0
python ../fx_gen.py --dsout datasets/example5_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example5_train.csv --modelout models/example5.pth \
--hlayers 200 200 --hactivations 'Tanh()' 'ReLU()' \
--loss 'SmoothL1Loss()' \
--optimizer 'RMSprop()' \
--epochs 150 --batch_size 100  \
--device cpu

python ../fx_gen.py --dsout datasets/example5_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example5.pth --ds datasets/example5_test.csv --predicted predictions/example5_pred.csv

python ../fx_plot.py --ds datasets/example5_test.csv --predicted predictions/example5_pred.csv
