#!/bin/bash

FX="np.sin(x)"
RB=-6.0
RE=6.0
python ../fx_gen.py --dsout datasets/example2_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example2_train.csv --modelout models/example2.pth \
  --epochs 250 --batch_size 100  \
  --optimizer 'Adam(lr=0.05, eps=1e-07)'

python ../fx_gen.py --dsout datasets/example2_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example2.pth --ds datasets/example2_test.csv --predictionout predictions/example2_pred.csv

python ../fx_plot.py --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv
#python ../fx_plot.py --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv --savefig predictions/example2.png
