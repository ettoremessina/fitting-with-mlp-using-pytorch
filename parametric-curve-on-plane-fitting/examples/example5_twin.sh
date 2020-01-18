#!/bin/bash

#Lissajous

FXT="2 * np.sin(0.5 * t + 1)"
FYT="3 * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example5_twin_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.01
python ../pmc2t_fit_twin.py --trainds datasets/example5_twin_train.csv --modelout models/example5_twin.pth \
  --hlayers 150 150 --hactivation 'ReLU()' 'ReLU()' \
  --epochs 400 \
  --optimizer 'Adamax(lr=0.01)'

python ../pmc2t_gen.py --dsout datasets/example5_twin_test.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.0475
python ../pmc2t_predict_twin.py --model models/example5_twin.pth --ds datasets/example5_twin_test.csv --predictionout predictions/example5_twin_pred.csv

python ../pmc2t_plot.py --ds datasets/example5_twin_test.csv --prediction predictions/example5_twin_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example5_twin_test.csv --prediction predictions/example5_twin_pred.csv --savefig predictions/example5_twin.png
