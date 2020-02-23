#!/bin/bash

FXY="x**2 + y**2"
RXB=-3.0
RXE=3.0
RYB=-3.0
RYE=3.0

python ../fxy_gen.py --dsout datasets/example1_train.csv --fxy "$FXY" --rxbegin $RXB --rxend $RXE --rybegin $RYB --ryend $RYE --rstep 0.05
python ../fxy_fit.py --trainds datasets/example1_train.csv --modelout models/example1.pth \
  --hlayers 120 160 --hactivations 'Tanh()' 'ReLU()' \
  --epochs 15 --batch_size 50 \
  --optimizer 'Adam(lr=0.05, eps=1e-07)' \
  --loss 'MSELoss()'

python ../fxy_gen.py --dsout datasets/example1_test.csv  --fxy "$FXY" --rxbegin $RXB --rxend $RXE --rybegin $RYB --ryend $RYE  --rstep 0.0875
python ../fxy_predict.py --model models/example1.pth --ds datasets/example1_test.csv --prediction predictions/example1_pred.csv

python ../fxy_plot.py --ds datasets/example1_test.csv --prediction predictions/example1_pred.csv
#python ../fxy_plot.py --ds datasets/example1_test.csv --prediction predictions/example1_pred.csv --savefig predictions/example1.png
