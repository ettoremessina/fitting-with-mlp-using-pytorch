# Parametric curve on plane fitting
This project implements the fitting of a continuous and limited real-valued parametric curve on plane where parameter belongs to a closed interval of the reals.
The curve fitting is implemented using a configurable multilayer perceptron neural network written using TensorFlow & Keras; it requires TensorFlow 2.0.0 library; it requires also NumPy and MatPlotLib libraries.

It contains four python programs:
 - **pmc2t_gen.py** generates a synthetic dataset file invoking a couple of one-variable real functions (one for x coordinate and the other one for y coordinate) on an real interval.
 - **pmc2t_fit.py** fits a parametric curve on plane in an interval using a configurable multilayer perceptron neural network.
 - **pmc2t_predict.py** makes a prediction on a test dataset of a parametric curve on place modeled with a pretrained multilayer perceptron neural network.
 - **pmc2t_plot.py** shows two overlapped x/y scatter graphs: the blue one is the dataset, the red one is the predicted one.

 ### Predefined examples of usage of the four command in cascade
 In the subfolder **examples** there are five shell scripts to fit five different parametric curves; each script executes the four programs in cascade in order to reach and show the goal.

```bash
$ cd parametric-curve-on-plane-fitting/examples
$ sh example1.sh
$ sh example2.sh
$ sh example3.sh
$ sh example4.sh
$ sh example5.sh
```

For details about the four commands and their command line options, please read below.


## pmc2t_gen.py
To get the usage of [pmc2t_gen.py](./pmc2t_gen.py) please run
```bash
$ python pmc2t_gen.py --help
```

and you get
```
usage: pmc2t_gen.py [-h]
  --dsout DS_OUTPUT_FILENAME
  --xt FUNCX_T_BODY
  --yt FUNCY_T_BODY
  [--rbegin RANGE_BEGIN]
  [--rend RANGE_END]
  [--rstep RANGE_STEP]

pmc2t_gen.py generates a synthetic dataset file of a parametric curve on plan
calling a couple of one-variable real functions in an interval

optional arguments:
  -h, --help            show this help message and exit
  --dsout DS_OUTPUT_FILENAME dataset output file (csv format)
  --xt FUNCX_T_BODY          x=fx(t) body (lamba format)
  --yt FUNCY_T_BODY          y=fy(t) body (lamba format)
  --rbegin RANGE_BEGIN       begin range (default:-5.0)
  --rend RANGE_END           end range (default:+5.0)
  --rstep RANGE_STEP         step range (default: 0.01)
```

where:
- **-h or --help** shows the above usage
- **--rbegin** and **--rend** are the limit of the closed interval of reals of independent parameter t.
- **--rstep** is the increment step of independent parameter t into interval.
- **--xt** is the function to use to compute the value of dependent variable x=fx(t); it is in lamba body format.
- **--yt** is the function to use to compute the value of dependent variable y=fy(t); it is in lamba body format.
- **--dsout** is the target dataset file name. The content of this file is csv and each line contains a triple of real numbers: the t, the fx(t) and the fy(t) where t is a value of the interval and fx(t) and fy(t) are the values of dependent variables; the dataset is sorted by independent variable t. This argument is mandatory.

### Example of pmc2t_gen.py usage
```bash
$ python pmc2t_gen.py --dsout mydataset.csv  --xt "0.1 * t * np.cos(t)" --yt "0.1 * t * np.sin(t)" --rbegin 0 --rend 20 --rstep 0.01
```


## pmc2t_fit.py
To get the usage of [pmc2t_fit.py](./pmc2t_fit.py) please run
```bash
$ python pmc2t_fit.py --help
```

and you get
```
usage: pmc2t_fit.py [-h]
  --trainds TRAIN_DATASET_FILENAME
  --modelout MODEL_PATH
  [--epochs EPOCHS]
  [--batch_size BATCH_SIZE]
  [--learning_rate LEARNING_RATE]
  [--hlayers HIDDEN_LAYERS]
  [--hunits HIDDEN_UNITS]
  [--hactivation HIDDEN_ACTIVATION]
  [--optimizer OPTIMIZER_NAME]
  [--decay DECAY]
  [--momentum MOMENTUM] [--nesterov NESTEROV]
                    [--epsilon EPSILON] [--rho RHO] [--beta_1 BETA_1]
                    [--beta_2 BETA_2] [--amsgrad AMSGRAD] [--loss LOSS]

pmc2t_fit.py fits a parametric curve on plan dataset using a configurable
multilayer perceptron network

optional arguments:
  -h, --help            show this help message and exit
  --trainds TRAIN_DATASET_FILENAME
                        train dataset file (csv format))
  --modelout MODEL_PATH
                        output model path
  --epochs EPOCHS       number of epochs)
  --batch_size BATCH_SIZE
                        batch size)
  --learning_rate LEARNING_RATE
                        learning rate)
  --hlayers HIDDEN_LAYERS
                        number of hidden layers
  --hunits HIDDEN_UNITS
                        number of neuors in each hidden layers
  --hactivation HIDDEN_ACTIVATION
                        activation function in hidden layers
  --optimizer OPTIMIZER_NAME
                        optimizer algorithm name
  --decay DECAY         decay
  --momentum MOMENTUM   momentum (used only by SGD optimizer)
  --nesterov NESTEROV   nesterov (used only by SGD optimizer)
  --epsilon EPSILON     epsilon (ignored by SGD optimizer)
  --rho RHO             rho (used only by RMSprop and Adadelta optimizers)
  --beta_1 BETA_1       beta_1 (used only by Adam, Adamax and Nadam
                        optimizers)
  --beta_2 BETA_2       beta_2 (used only by Adam, Adamax and Nadam
                        optimizers)
  --amsgrad AMSGRAD     amsgrad (used only by Adam optimizer)
  --loss LOSS           loss function name
```
