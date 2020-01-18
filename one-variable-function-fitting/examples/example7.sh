#!/bin/bash

FX="np.exp(np.sin(x))"
RB=-5.0
RE=5.0

python ../fx_gen.py --dsout datasets/example7_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example7_train.csv --modelout models/example7.pth --epochs 500 --batch_size 100 --optimizer 'Adam(lr=0.01)' 


python ../fx_gen.py --dsout datasets/example7_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example7.pth --ds datasets/example7_test.csv --predictionout predictions/example7_pred.csv

python ../fx_plot.py --ds datasets/example7_test.csv --prediction predictions/example7_pred.csv
