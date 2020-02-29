# Parametric curve in space fitting
This project implements the fitting of a continuous and limited real-valued parametric curve in space where parameter belongs to a closed interval of the reals.
The curve fitting is implemented using a configurable multilayer perceptron neural network written using PyTorch 1.2.0; it requires also NumPy and MatPlotLib libraries.<br />

Please visit [here](https://computationalmindset.com/en/neural-networks/parametric-curve-in-space-fitting-with-pytorch.html) for concepts about this project.

It contains four python programs:
- **pmc3t_gen.py** generates a synthetic dataset file invoking a triple of one-variable real functions defined on an real interval: first one for x=x(t) coordinate, second one for y=y(t) coordinate and third one for z=z(t) coordinate.
- **pmc3t_fit.py** fits a parametric curve in space using a configurable multilayer perceptron in order to fit a vector function f(t) = [x(t), y(t), z(t)].
- **pmc3t_predict.py** makes a prediction on a parametric curve in space modeled with a pretrained multilayer perceptron.
- **pmc3t_plot.py** shows two overlapped x/y/z scatter graphs: the blue one is the input dataset, the red one is the prediction.


### Predefined examples of usage of the four command in cascade
In the subfolder **examples** there are five shell scripts to fit five different parametric curves; each script executes the four programs in cascade in order to reach and show the goal.

```bash
$ cd parametric-curve-in-space-fitting/examples
$ sh example1.sh
$ sh example2.sh
$ sh example3.sh
$ sh example4.sh
$ sh example5.sh
```

For details about the commands and their command line options, please read below.


## pmc3t_gen.py<a name="pmc3t_gen"/>
To get the usage of [pmc3t_gen.py](./pmc3t_gen.py) please run:
```bash
$ python pmc3t_gen.py --help
```

and you get:
```
usage: pmc3t_gen.py [-h]
  --dsout DS_OUTPUT_FILENAME
  --xt FUNCX_T_BODY
  --yt FUNCY_T_BODY
  --zt FUNCY_T_BODY
  [--rbegin RANGE_BEGIN]
  [--rend RANGE_END]
  [--rstep RANGE_STEP]

pmc3t_gen.py generates a synthetic dataset file that contains the points of a parametric curve in space
calling a triple of one-variable real functions in an interval

optional arguments:
  -h, --help            show this help message and exit
  --dsout DS_OUTPUT_FILENAME dataset output file (csv format)
  --xt FUNCX_T_BODY          x=x(t) body (lamba format)
  --yt FUNCY_T_BODY          y=y(t) body (lamba format)
  --zt FUNCY_T_BODY          z=z(t) body (lamba format)
  --rbegin RANGE_BEGIN       begin range (default:-5.0)
  --rend RANGE_END           end range (default:+5.0)
  --rstep RANGE_STEP         step range (default: 0.01)
```

Namely:
- **-h or --help** shows the above usage
- **--rbegin** and **--rend** are the limit of the closed interval of reals of independent parameter t.
- **--rstep** is the increment step of independent parameter t into interval.
- **--xt** is the function to use to compute the value of dependent variable x=x(t); it is in lamba body format.
- **--yt** is the function to use to compute the value of dependent variable y=y(t); it is in lamba body format.
- **--zt** is the function to use to compute the value of dependent variable z=z(t); it is in lamba body format.
- **--dsout** is the target dataset file name. The content of this file is csv (no header at first line) and each line contains a triple of real numbers: t, x(t) and y(t) where t is a value of the interval and x(t) and y(t) are the values of dependent variables. This argument is mandatory.

### Example of pmc3t_gen.py usage
```bash
$ python pmc3t_gen.py --dsout mydataset.csv  --xt "0.1 * t * np.cos(t)" --yt "0.1 * t * np.sin(t)" --zt "t" --rbegin 0 --rend 20 --rstep 0.01
```


## pmc3t_fit.py<a name="pmc3t_fit"/>
To get the usage of [pmc3t_fit.py](./pmc3t_fit.py) please run:
```bash
$ python pmc3t_fit.py --help
```

and you get:
```
usage: pmc3t_fit.py [-h] --trainds
                    TRAIN_DATASET_FILENAME
                    --modelout MODEL_PATH
                    [--epochs EPOCHS]
                    [--batch_size BATCH_SIZE]
                    [--hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...]]
                    [--hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...]]
                    [--optimizer OPTIMIZER]
                    [--loss LOSS]
                    [--device DEVICE]

pmc3t_fit.py fits a parametric curve in space dataset using a configurable
multilayer perceptron with three output neurons

optional arguments:
  -h, --help                       show this help message and exit
  --trainds TRAIN_DATASET_FILENAME train dataset file (csv format)
  --modelout MODEL_PATH            output model file
  --epochs EPOCHS                  number of epochs
  --batch_size BATCH_SIZE          batch size
  --hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...] number of neurons for each hidden layers
  --hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...] activation functions between layers
  --optimizer OPTIMIZER            optimizer algorithm object
  --loss LOSS                      loss function name
  --device DEVICE                  target device
```

Namely:
- **-h or --help** shows the above usage
- **--trainds** is the input training dataset in csv format: four real numbers for each line respectively for x and y (no header at first line). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **pmc3t_gen.py**. This argument is mandatory.
- **--modelout** is a non-existing file where the program saves the trained model (in pth format). This argument is mandatory.
- **--epochs** is the number of epochs of the training process. The default is **500**
- **--batch_size** is the size of the batch used during training. The default is **50**
- **--hlayers** is a sequence of integers: the size of the sequence is the number of hidden layers, each value of the sequence is the number of neurons in the correspondent layer. The default is **100** (that means one only hidden layer with 100 neurons)
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

### Examples of pmc3t_fit.py usage
```bash
$ python pmc3t_fit.py \
  --trainds mytrainds.csv \
  --modelout mymodel.pth \
  --hlayers 200 300 200 \
  --hactivation 'Sigmoid()' 'Tanh()' 'Sigmoid()' \
  --epochs 250

$ python pmc3t_fit.py \
  --trainds mytrainds2.csv \
  --modelout mymodel2.pth \
  --hlayers 200 300 200 \
  --hactivation 'Sigmoid()' 'Tanh()' 'Sigmoid()' \
  --epochs 250

$ python pmc3t_fit.py \
      --trainds mytrainds.csv \
      --modelout mymodel.pth \
      --hlayers 200 200 200 \
      --hactivation 'Sigmoid()' 'Sigmoid()' 'Sigmoid()' \
      --epochs 250 \
      --optimizer 'Adamax(lr=1.1e-2)'

$ python pmc3t_fit.py \
    --trainds mytraintds2.csv \
    --modelout mymodel2.pth \
    --hlayers 200 200 200 \
    --hactivation 'Sigmoid()' 'Sigmoid()' 'Sigmoid()' \
    --epochs 250 \
    --optimizer 'Adamax(lr=1.1e-2)'
```


## pmc3t_predict.py<a name="pmc3t_predict"/>
To get the usage of [pmc3t_predict.py](./pmc3t_predict.py) please run:
```bash
$ python pmc3t_predict.py --help
```

and you get:
```
usage: pmc3t_predict.py [-h]
                        --model MODEL_PATH
                        --ds DATASET_FILENAME
                        --predictionout PREDICTION_DATA_FILENAME
                        [--device DEVICE]

pmc3t_predict.py makes prediction of couples of coordinates of a parametric curve
in space modeled with a pretrained multilayer perceptron with three output neurons

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL_PATH    model file
  --ds DATASET_FILENAME dataset file (csv format); only t-values are used
  --predictionout PREDICTION_DATA_FILENAME prediction data file (csv format)
  --device DEVICE       target device
```

Namely:
- **-h or --help** shows the above usage
- **--model** is the pth file of a model generated by **pmc3t_fit.py** (see **--modelout** command line parameter of **pmc3t_fit.py**). This argument is mandatory.
- **--ds** is the input dataset in csv format (no header at first line): program uses only the x values (first column). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **pmc3t_gen.py**. This argument is mandatory.
- **--predictionout** is the file name of prediction values. The content of this file is csv (no header at first line) and each line contains a triple of real numbers: the t value comes from input dataset and the prediction is the couple of values of x(t) and y(t) computed by multilayer perceptron model on t value; this argument is mandatory.
- **--device** is the target CUDA device where to perform math computations; default is **cpu**; to use default GPU pass **cuda**; please see [PyTorch CUDA semantic reference](https://pytorch.org/docs/stable/notes/cuda.html) for details.

### Example of pmc3t_predict.py usage
```bash
$ python pmc3t_predict.py --model mymodel.pth --ds mytestds.csv --predictionout myprediction.csv
```


## pmc3t_plot.py<a name="pmc3t_plot"/>
To get the usage of [pmc3t_plot.py](./pmc3t_plot.py) please run:
```bash
$ python pmc3t_plot.py --help
```

and you get:
```
usage: pmc3t_plot.py [-h]
                     --ds DATASET_FILENAME
                     --prediction PREDICTION_DATA_FILENAME
                     [--savefig SAVE_FIGURE_FILENAME]

pmc3t_plot.py shows two overlapped x/y scatter graphs: the blue one is the
dataset, the red one is the prediction

optional arguments:
  -h, --help                     show this help message and exit
  --ds DATASET_FILENAME          dataset file (csv format)
  --prediction PREDICTION_DATA_FILENAME prediction data file (csv format)
  --savefig SAVE_FIGURE_FILENAME if present, the chart is saved on a file instead to be shown on screen
```

Namely:
- **-h or --help** shows the above usage
- **--ds** is an input dataset in csv format (no header at first line). Usually this parameter is the test dataset file passed to **pmc3t_predict.py**, but you could pass the training dataset passed to **pmc3t_fit.py**. This argument is mandatory.
- **--prediction** is the file name of prediction values generated by **pmc3t_predict.py** (see **--predictionout** command line parameter of **pmc3t_predict.py**). This argument is mandatory.
- **--savefig** if this argument is missing, the chart is shown on screen, otherwise this argument is the png output filename where **pmc3t_plot.py** saves the chart.

### Examples of pmc3t_plot.py usage
```bash
$ python pmc3t_plot.py --ds mytestds.csv --prediction myprediction.csv

$ python pmc3t_plot.py --ds mytrainds.csv --prediction myprediction.csv --savefig mychart.png
```
