# One variable function fitting
This project implements the fitting of a continuous and limited real-valued function defined in a closed interval of the reals.
The function fitting is implemented using a configurable multilayer perceptron neural network written using PyTorch 1.2.0; it requires also NumPy and MatPlotLib libraries.

It contains four python programs:
 - **fx_gen.py** generates a synthetic dataset file invoking a one-variable real function on an real interval.
 - **fx_fit.py** fits a one-variable function in an interval using a configurable multilayer perceptron neural network.
 - **fx_predict.py** makes a prediction on a test dataset of a one-variable function modeled with a pretrained multilayer perceptron neural network.
 - **fx_plot.py** shows two overlapped x/y scatter graphs: the blue one is the train dataset, the red one is the predicted one.

### Predefined examples of usage of the four command in cascade
In the subfolder **examples** there are nine bash scripts to fit nine different one-variable functions; each script executes the four programs in cascade in order to reach and show the goal.

```bash
$ cd one-variable-function-fitting/examples
$ sh example1.sh
$ sh example2.sh
$ sh example3.sh
$ sh example4.sh
$ sh example5.sh
$ sh example6.sh
$ sh example7.sh
$ sh example8.sh
$ sh example9.sh
```

For details about the four commands and their command line options, please read below.


## fx_gen.py
To get the usage of [fx_gen.py](./fx_gen.py) please run
```bash
$ python fx_gen.py --help
```

and you get
```
fx_gen.py generates a synthetic dataset file calling a one-variable real function in an interval

optional arguments:
  -h, --help                  show this help message and exit
  --dsout DS_OUTPUT_FILENAME  dataset output file (csv format)
  --fx FUNC_X_BODY            f(x) body (body lamba format)
  --rbegin RANGE_BEGIN        begin range (default:-5.0)
  --rend RANGE_END            end range (default:+5.0)
  --rstep RANGE_STEP          step range (default: 0.01)
```

where:
- **-h or --help** shows the above usage
- **--rbegin** and **--rend** are the limit of the closed interval of reals of independent variable x.
- **--rstep** is the increment step of independent variable x into interval.
- **--fx** is the function to use to compute the value of dependent variable; it is in lamba body format.
- **--dsout** is the target dataset file name. The content of this file is csv and each line contains a couple of real numbers: the x and the f(x) where x is a value of the interval and f(x) is the value of dependent variable; the dataset is sorted by independent variable x. This argument is mandatory.

### Examples of fx_gen.py usage
```bash
$ python fx_gen.py --dsout mydataset.csv  --fx "np.exp(np.sin(x))" --rbegin -6.0 --rend 6.0 --rstep 0.05

$ python fx_gen.py --dsout mydataset.csv  --fx "np.sqrt(np.abs(x))" --rbegin -5.0 --rend 5.0 --rstep 0.04
```


## fx_fit.py
To get the usage of [fx_fit.py](./fx_fit.py) please run
```bash
$ python fx_fit.py --help
```

and you get
```
usage: fx_fit.py [-h] --trainds TRAIN_DATASET_FILENAME --modelout MODEL_PATH
                 [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                 [--learning_rate LEARNING_RATE]
                 [--hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...]]
                 [--hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...]]
                 [--optimizer OPTIMIZER] [--loss LOSS] [--device DEVICE]

fx_fit.py fits a one-variable function in an interval using a configurable
multilayer perceptron network implemented in PyTorch

optional arguments:
  -h, --help                        show this help message and exit
  --trainds TRAIN_DATASET_FILENAME  train dataset file (csv format)
  --modelout MODEL_PATH             output model file
  --epochs EPOCHS                   number of epochs
  --batch_size BATCH_SIZE           batch size
  --learning_rate LEARNING_RATE     learning rate
  --hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...] number of neurons for each hidden layers
  --hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...] activation functions between layers
  --optimizer OPTIMIZER             optimizer algorithm object
  --loss LOSS                       loss function name
  --device DEVICE                   target device
```

where:
- **-h or --help** shows the above usage
- **--trainds** is the input training dataset in csv format: a couple of real number for each line respectively for x and y (no header in first line). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **fx_gen.py**. This argument is mandatory.
- **--modelout** is a non-existing file where the program saves the trained model (in pth format). This argument is mandatory.
- **--epochs** is the number of epochs of the training process. The default is **500**
- **--batch_size** is the size of the batch used during training. The default is **50**
- **--learning_rate** is the learning rate. The default depends by the chosen optimizer (see below) in according with [PyTorch optimizer algorithm reference](https://pytorch.org/docs/stable/optim.html#algorithms).\
**Note:** the learning rate can be passed either via **--learning_rate** command line argument or via **lr** named parameter of constructor (see below), but never in both way.
- **--hlayers** is a sequence of integers: the size of the sequence is the number of hidden layers, each value of the sequence is the number of neurons in the correspondent layer. The default is **100** (one only hidden layer with 100 neurons),
- **--hactivations** is a sequence of activation function constructor calls: the size of the sequence must be equal to the number of layers and each item of the sequence is the activation function to apply to the output of the neurons of the correspondent layer; please see [PyTorch activation function reference](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) for details about constructor named parameters and examples at the end of this section.\
  Available activation function constructors are:
  - ELU()
  - Hardshrink()
  - Hardtanh()
  - LeakyReLU()
  - LogSigmoid()
  - MultiheadAttention()
  - PReLU()
  - ReLU()
  - ReLU6()
  - RReLU()
  - SELU()
  - CELU()
  - Sigmoid()
  - Softplus()
  - Softshrink()
  - Softsign()
  - Tanh()
  - Tanhshrink()
  - Threshold() \
  The default is **ReLU()** (applied to one only hidden layer; if number of layers are > 1, this argument becomes mandatory).
- **--optimizer** is the constructor call of the algorithm used by the training process. You can pass also named arguments between round brackets; please see [PyTorch optimizer algorithm reference](https://pytorch.org/docs/stable/optim.html#algorithms) for details about constructor named parameters and examples at the end of this section.\
  Available algorithm constructors are:
  - Adadelta()
  - Adagrad()
  - Adam()
  - AdamW()
  - ASGD()
  - LBFGS()
  - SparseAdam()
  - RMSprop()
  - Rprop()
  - SGD()\
  The default is **Adam()**.
- **--loss** is the constructor call of the loss function used by the training process. You can pass also named arguments between round brackets; please see [PyTorch loss functions reference](https://pytorch.org/docs/stable/nn.html#loss-functions) for details about constructor named parameters and examples at the end of this section.\
  Available loss function construtors are:
  - L1Loss()
  - MSELoss()
  - CrossEntropyLoss()
  - CTCLoss()
  - NLLLoss()
  - PoissonNLLLoss()
  - KLDivLoss()
  - BCELoss()
  - BCEWithLogitsLoss()
  - MarginRankingLoss()
  - HingeEmbeddingLoss()
  - MultiLabelMarginLoss()
  - SmoothL1Loss()
  - SoftMarginLoss()
  - MultiLabelSoftMarginLoss()
  - CosineEmbeddingLoss()
  - MultiMarginLoss()
  - TripletMarginLoss()\
  The default is **MSELoss()**.
  - **--device** is the target CUDA device where to perform math computations; default is **cpu**; to use default GPU pass **cuda**; please see [Pytorch CUDA semantic reference for details](https://pytorch.org/docs/stable/notes/cuda.html) for advances details.

### Examples of fx_fix.py usage
```bash
$ python fx_fit.py \
  --trainds mytrainds.csv \
  --modelout mymodel.pth \
  --hlayers 200 200 \
  --hactivation 'ReLU()' 'ReLU()' \
  --epochs 500 --batch_size 100

$ python fx_fit.py \
  --trainds mytrainds.csv \
  --modelout mymodel.pth \
  --hlayers 120 160 \
  --hactivations tanh relu \
  --epochs 100 \
  --batch_size 50 \
  --optimizer 'Adam(eps=1e-07)' \
  --learning_rate 0.05 \
  --loss 'MSELoss()'

$ python fx_fit.py \
  --trainds mytrainds.csv
  --modelout mymodel.pth \
  --hlayers 200 300 200 \
  --hactivation 'Sigmoid()' 'Sigmoid()' 'Sigmoid()' \
  --epochs 1000 \
  --batch_size 200 \
  --optimizer 'Adamax()' \
  --learning_rate 0.02

$ python fx_fit.py \
  --trainds mytrainds.csv
  --modelout mymodel.pth \
  --hlayers 200 300 200 \
  --hactivation 'Sigmoid()' 'Sigmoid()' 'Sigmoid()' \
  --epochs 1000 \
  --batch_size 200 \
  --optimizer 'Adamax(lr=0.02)' \
  --device cuda
```


## fx_predict.py
To get the usage of [fx_predict.py](./fx_predict.py) please run
```bash
$ python fx_predict.py --help
```

and you get
```
usage: fx_predict.py [-h]
                     --model MODEL_PATH
                     --testds TEST_DATASET_FILENAME
                     --predictedout PREDICTED_DATA_FILENAME
                     [--device DEVICE]

fx_predict.py makes a prediction on a test dataset of a one-variable function
modeled with a pretrained multilayer perceptron network

optional arguments:
  -h, --help                              show this help message and exit
  --model MODEL_PATH                      model path
  --testds TEST_DATASET_FILENAME          test dataset file (csv format)
  --predictedout PREDICTED_DATA_FILENAME  predicted data file (csv format)
  --device DEVICE                         target device
```

where:
- **-h or --help** shows the above usage
- **--model** is the pth file of a model generated by **fx_fit.py** (see **--modelout** command line parameter of **fx_fit.py**). This argument is mandatory.
- **--testds** is the input test dataset in csv format: a couple of real number for each line respectively for x and y (no header in first line). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **fx_gen.py**. This argument is mandatory.
- **--predictedout** is the file name of predicted values. The content of this file is csv and each line contains a couple of real numbers: the x value come from test dataset and the predicted f(x); This argument is mandatory.
- **--device** is the target CUDA device where to perform math computations; default is **cpu**; to use default GPU pass **cuda**; please see [Pytorch CUDA semantic reference for details](https://pytorch.org/docs/stable/notes/cuda.html) for advances details.

### Example of fx_predict.py usage
```bash
$ python fx_predict.py --model mymodel.pth --testds mytestds.csv --predictedout myprediction.csv
```


## fx_plot.py
To get the usage of [fx_plot.py](./fx_plot.py) please run
```bash
$ python fx_plot.py --help
```

and you get
```
usage: fx_plot.py [-h]
                  --trainds TRAIN_DATASET_FILENAME
                  --predicted PREDICTED_DATA_FILENAME
                  [--savefig SAVE_FIGURE_FILENAME]

fx_plot.py shows two overlapped x/y scatter graphs: the blue one is the train
dataset, the red one is the predicted one

optional arguments:
  -h, --help            show this help message and exit
  --trainds TRAIN_DATASET_FILENAME     train dataset file (csv format)
  --predicted PREDICTED_DATA_FILENAME  predicted data file (csv format)
  --savefig SAVE_FIGURE_FILENAME       if present, the chart is saved on a file instead to be shown on screen
```
where:
- **-h or --help** shows the above usage
- **--trainds** is the input training dataset in csv format passed before to **fx_fit.py** (see **--trainds** command line parameter of **fx_fit.py**). This argument is mandatory.
- **--predicted** is the file name of predicted values generated by **fx_predict.py** (see **--predictedout** command line parameter of **fx_predict.py**). This argument is mandatory.
- **--savefig** if this argument is missing, the chart is shown on screen, otherwise this argument is the png output filename where **fx_plot.py** saves the chart.

### Example of fx_plot.py usage
```bash
$ python fx_plot.py --trainds mytrainds.csv --predicted myprediction.csv

$ python fx_plot.py --trainds mytrainds.csv --predicted myprediction.csv --savefig mychart.png
```