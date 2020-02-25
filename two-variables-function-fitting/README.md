# Two-variables real-valued function fitting
This project implements the fitting of a continuous and limited real-valued function of two variables constrained in a rectangle.
This two-variables real-valued function fitting is implemented using a configurable multilayer perceptron neural network written using PyTorch 1.2.0; it requires also NumPy and MatPlotLib libraries.<br/>

It contains four python programs:
 - **fxy_gen.py** generates a synthetic dataset file invoking a two-variables real-valued function constrained in a rectangle.
 - **fxy_fit.py** fits a two-variables real-valued function constrained in a rectangle using a configurable multilayer perceptron.
 - **fxy_predict.py** makes a prediction of a two-variables real-valued function modeled with a pretrained multilayer perceptron.
 - **fxy_plot.py** shows two non overlapped x/y/z scatter graphs: the blue one is the input dataset, the red one is the prediction.

### Predefined examples of usage of the four command in cascade
In the subfolder **examples** there are three shell scripts to fit three different two-variables real-valued functions; each script executes the four programs in cascade in order to reach and show the goal.

```bash
$ cd two-variables-function-fitting/examples
$ sh example1.sh
$ sh example2.sh
$ sh example3.sh
```

For details about the four commands and their command line options, please read below.


## fxy_gen.py<a name="fxy_gen"/>
To get the usage of [fxy_gen.py](./fxy_gen.py) please run:
```bash
$ python fxy_gen.py --help
```

and you get:
```
usage: fxy_gen.py [-h]
                 --dsout DS_OUTPUT_FILENAME
                 --fxy FUNC_XY_BODY
                 [--rxbegin RANGE_XBEGIN] [--rxend RANGE_XEND]
                 [--rybegin RANGE_YBEGIN] [--ryend RANGE_YEND]
                 [--rstep RANGE_STEP]

fxy_gen.py generates a synthetic dataset file calling a two-variables real-valued function constrained in a rectangle.

optional arguments:
  -h, --help                 show this help message and exit
  --dsout DS_OUTPUT_FILENAME dataset output file (csv format)
  --fxy FUNC_XY_BODY         f(x,y) body (lamba format)
  --rxbegin RANGE_XBEGIN     begin x range (default:-5.0)
  --rxend RANGE_XEND         end x range (default:+5.0)
  --rybegin RANGE_YBEGIN     begin y range (default:-5.0)
  --ryend RANGE_YEND         end y range (default:+5.0)
  --rstep RANGE_STEP         step range (default: 0.01)
```

Namely:
- **-h or --help** shows the above usage
- **--rxbegin** and **--rxend** are the limit of the closed interval of reals of independent variable x.
- **--rybegin** and **--ryend** are the limit of the closed interval of reals of independent variable y.
- **--rstep** is the incremental step of independent variable x into the interval.
- **--fxy** is the two-variables real-value function to use to compute the value of dependent variable; it is in lamba body format.
- **--dsout** is the target dataset file name. The content of this file is csv (no header at first line) and each line contains a triple of real numbers: the x, the y and the f(x, y) where x is a value of the interval [rxbegin, rxend], y is a value in the interval [rybegin, ryend] and f(x, y) is the value of dependent variable. This argument is mandatory.

### Examples of fxy_gen.py usage
```bash
$ python fxy_gen.py --dsout mydataset.csv  --fxy "x**2 + y**2" --rxbegin -3.0 --rxend 3.0  --rybegin -3.0 --ryend 3.0 --rstep 0.05

$ python fxy_gen.py --dsout mydataset.csv  --fxy "np.sin(np.sqrt(x**2 + y**2))" --rxbegin -5.0 --rxend 5.0  --rybegin -5.0 --ryend 5.0 --rstep 0.04
```


## fxy_fit.py<a name="fxy_fit"/>
To get the usage of [fxy_fit.py](./fxy_fit.py) please run:
```bash
$ python fxy_fit.py --help
```

and you get:
```
usage: fxy_fit.py [-h]
                 --trainds TRAIN_DATASET_FILENAME
                 --modelout MODEL_PATH
                 [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                 [--hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...]]
                 [--hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...]]
                 [--optimizer OPTIMIZER]
                 [--loss LOSS]
                 [--device DEVICE]

fxy_fit.py fits a two-variables real-valued function dataset using a configurable multilayer perceptron

optional arguments:
  -h, --help                        show this help message and exit
  --trainds TRAIN_DATASET_FILENAME  train dataset file (csv format)
  --modelout MODEL_PATH             output model file
  --epochs EPOCHS                   number of epochs
  --batch_size BATCH_SIZE           batch size
  --hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...] number of neurons for each hidden layer
  --hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...] activation functions between layer
  --optimizer OPTIMIZER             optimizer algorithm
  --loss LOSS                       loss function
  --device DEVICE                   target device
```

Namely:
- **-h or --help** shows the above usage
- **--trainds** is the input training dataset in csv format: a triple of real numbers for each line respectively for x, y and z (no header at first line). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **fxy_gen.py**. This argument is mandatory.
- **--modelout** is a non-existing file where the program saves the trained model (in pth format). This argument is mandatory.
- **--epochs** is the number of epochs of the training process. The default is **500**
- **--batch_size** is the size of the batch used during training. The default is **50**
- **--hlayers** is a sequence of integers: the size of the sequence is the number of hidden layers, each value of the sequence is the number of neurons in the correspondent layer. The default is **100** (that means one only hidden layer with 100 neurons),
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
  The default is **ReLU()** (if number of layers is > 1, this argument becomes mandatory).
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
  **Note:** an important parameter often passed to optimizer constructor is the learning rate: in order to pass a value for learning rate you can use **lr** named parameter of constructor.
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
  - **--device** is the target CUDA device where to perform math computations; default is **cpu**; to use default GPU pass **cuda**; please see [PyTorch CUDA semantic reference](https://pytorch.org/docs/stable/notes/cuda.html) for details.

### Examples of fxy_fix.py usage
```bash
$ python fxy_fit.py 
  --trainds mytrain.csv \
  --modelout mymodels \
  --hlayers 120 160 \
  --hactivations tanh relu \
  --epochs 15 \
  --batch_size 50 \
  --optimizer 'Adam(learning_rate=0.05, epsilon=1e-07)' \
  --loss 'MeanSquaredError()'

$ python fxy_fit.py \
  --trainds mytrain.csv \
  --modelout mymodels \
  --hlayers 100 100 \
  --hactivations relu relu \
  --epochs 10 \
  --batch_size 50 \
  --optimizer 'SGD(decay=1e-6, momentum=0.9, nesterov=True)' \
  --loss 'MeanSquaredError()'

$ python fxy_fit.py \
  --trainds mytrain.csv \
  --modelout mymodels \
  --hlayers 200 300 200 \
  --hactivations tanh tanh tanh \
  --epochs 20 \
  --batch_size 100 \
  --optimizer 'Adamax(lr=0.01)' \
  --loss 'MeanSquaredError()' \
  --device cuda
```


## fxy_predict.py<a name="fxy_predict"/>
To get the usage of [fxy_predict.py](./fxy_predict.py) please run
```bash
$ python fxy_predict.py --help
```

and you get:
```
usage: fxy_predict.py [-h]
                     --model MODEL_PATH
                     --ds TEST_DATASET_FILENAME
                     --predictionout PREDICTION_DATA_FILENAME
                     [--device DEVICE]

fxy_predict.py makes prediction of the values of a two-variables real-valued function modeled with a pretrained multilayer perceptron

optional arguments:
  -h, --help                               show this help message and exit
  --model MODEL_PATH                       model file
  --ds DATASET_FILENAME                    input dataset file (csv format); only x-values are used
  --predictionout PREDICTION_DATA_FILENAME  prediction data file (csv format)
  --device DEVICE                         target device
```

Namely:
- **-h or --help** shows the above usage
- **--model** is the pth file of a model generated by **fxy_fit.py** (see **--modelout** command line parameter of **fxy_fit.py**). This argument is mandatory.
- **--ds** is the input dataset in csv format (no header at first line): program uses only the x values (first column). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **fxy_gen.py**. This argument is mandatory.
- **--predictionout** is the file name of prediction values. The content of this file is csv (no header at first line) and each line contains a couple of real numbers: the x value comes from input dataset and the prediction is the value of f(x) computed by multilayer perceptron model on x value; this argument is mandatory.
- **--device** is the target CUDA device where to perform math computations; default is **cpu**; to use default GPU pass **cuda**; please see [PyTorch CUDA semantic reference](https://pytorch.org/docs/stable/notes/cuda.html) for details.

### Example of fxy_predict.py usage
```bash
$ python fxy_predict.py --model mymodel.pth --ds mytestds.csv --predictionout myprediction.csv
```


## fxy_plot.py<a name="fxy_plot"/>
To get the usage of [fxy_plot.py](./fxy_plot.py) please run
```bash
$ python fxy_plot.py --help
```

and you get:
```
usage: fxy_plot.py [-h]
                  --ds DATASET_FILENAME
                  --prediction PREDICTION_DATA_FILENAME
                  [--savefig SAVE_FIGURE_FILENAME]

fxy_plot.py shows two non overlapped x/y/z scatter graphs: the blue one is the dataset, the red one is the prediction one

optional arguments:
  -h, --help            show this help message and exit
  --ds DATASET_FILENAME dataset file (csv format)
  --prediction PREDICTION_DATA_FILENAME  prediction data file (csv format)
  --savefig SAVE_FIGURE_FILENAME       if present, the chart is saved on a file instead to be shown on screen
```

Namely:
- **-h or --help** shows the above usage
- **--ds** is an input dataset in csv format (no header at first line). Usually this parameter is the test dataset file passed to **fxy_predict.py**, but you could pass the training dataset passed to **fxy_fit.py**. This argument is mandatory.
- **--prediction** is the file name of prediction values generated by **fxy_predict.py** (see **--predictionout** command line parameter of **fyx_predict.py**). This argument is mandatory.
- **--savefig** if this argument is missing, the chart is shown on screen, otherwise this argument is the png output filename where **fxy_plot.py** saves the chart.

### Examples of fxy_plot.py usage
```bash
$ python fxy_plot.py --ds mytestds.csv --prediction myprediction.csv

$ python fxy_plot.py --ds mytrainds.csv --prediction myprediction.csv --savefig mychart.png
```
