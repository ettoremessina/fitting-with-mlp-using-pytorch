# Parametric curve on plane fitting
This project implements the fitting of a continuous and limited real-valued parametric curve on plane where parameter belongs to a closed interval of the reals.
The curve fitting is implemented using a configurable multilayer perceptron neural network written using PyTorch 1.2.0; it requires also NumPy and MatPlotLib libraries.<br />

Please visit [here](https://computationalmindset.com/en/posts/neural-networks/parametric-curve-on-plane-fitting-with-pytorch.html) for concepts about this project.

It contains four python programs:
- **pmc2t_gen.py** generates a synthetic dataset file invoking a pair of one-variable real functions defined on an real interval: first one for x=x(t) coordinate and the other one for y=y(t) coordinate.
- **pmc2t_fit.py** fits a parametric curve on plane using a configurable multilayer perceptron in order to fit a vector function f(t) = [x(t), y(t)].
- **pmc2t_predict.py** makes a prediction on a parametric curve on place modeled with a pretrained multilayer perceptron.
- **pmc2t_plot.py** shows two overlapped x/y scatter graphs: the blue one is the input dataset, the red one is the prediction.

The project contains also other two programs to implement the **twin** variant:
- **pmc2t_fit_twin.py** has same purpose of **pmc2t_fit.py** but it uses a configurable pair of twins of multilayer perceptrons in order to fix separately the one variable functions x=x(t) and y=y(t).
- **pmc2t_predict_twin.py** has same purpose of **pmc2t_predict.py** but it takes in input the pair of multilayer perceptron models trained by **pmc2t_fit_twin.py**.


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

In the subfolder **examples** there are also the same five shell scripts for the **twin** variant.

```bash
$ cd parametric-curve-on-plane-fitting/examples
$ sh example1_twin.sh
$ sh example2_twin.sh
$ sh example3_twin.sh
$ sh example4_twin.sh
$ sh example5_twin.sh
```

The **twin** variant spends double of time, so it is not nice from performance point of view. Anyway it could be interesting to compare the behavior of a multilayer perceptron that fits a vector function with the behavior of a pair of twins of multilayer perceptrons that fit separately the two component functions.
For details about the commands and their command line options, please read below.


## pmc2t_gen.py<a name="pmc2t_gen"/>
To get the usage of [pmc2t_gen.py](./pmc2t_gen.py) please run:
```bash
$ python pmc2t_gen.py --help
```

and you get:
```
usage: pmc2t_gen.py [-h]
  --dsout DS_OUTPUT_FILENAME
  --xt FUNCX_T_BODY
  --yt FUNCY_T_BODY
  [--rbegin RANGE_BEGIN]
  [--rend RANGE_END]
  [--rstep RANGE_STEP]

pmc2t_gen.py generates a synthetic dataset file that contains the points of a parametric curve on plane
calling a couple of one-variable real functions in an interval

optional arguments:
  -h, --help            show this help message and exit
  --dsout DS_OUTPUT_FILENAME dataset output file (csv format)
  --xt FUNCX_T_BODY          x=x(t) body (lamba format)
  --yt FUNCY_T_BODY          y=y(t) body (lamba format)
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
- **--dsout** is the target dataset file name. The content of this file is csv (no header at first line) and each line contains a triple of real numbers: t, x(t) and y(t) where t is a value of the interval and x(t) and y(t) are the values of dependent variables. This argument is mandatory.

### Example of pmc2t_gen.py usage
```bash
$ python pmc2t_gen.py --dsout mydataset.csv  --xt "0.1 * t * np.cos(t)" --yt "0.1 * t * np.sin(t)" --rbegin 0 --rend 20 --rstep 0.01
```


## pmc2t_fit.py and pmc2t_fit_twin.py<a name="pmc2t_fit"/>
To get the usage of [pmc2t_fit.py](./pmc2t_fit.py) please run:
```bash
$ python pmc2t_fit.py --help
```

and you get:
```
usage: pmc2t_fit.py [-h] --trainds
                    TRAIN_DATASET_FILENAME
                    --modelout MODEL_PATH
                    [--epochs EPOCHS]
                    [--batch_size BATCH_SIZE]
                    [--hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...]]
                    [--hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...]]
                    [--optimizer OPTIMIZER]
                    [--loss LOSS]
                    [--device DEVICE]

pmc2t_fit.py fits a parametric curve on plane dataset using a configurable
multilayer perceptron with two output neurons

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

To get the usage of [pmc2t_fit_twin.py](./pmc2t_fit_twin.py) please run:
```bash
$ python pmc2t_fit_twin.py --help
```

you will get the identical set of command line parameters of **pmc2t_fit.py**

Namely:
- **-h or --help** shows the above usage
- **--trainds** is the input training dataset in csv format: a triple of real numbers for each line respectively for x and y (no header at first line). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **pmc2t_gen.py**. This argument is mandatory.
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

### Examples of pmc2t_fit.py and pmc2t_fit_twin.py usage
```bash
$ python pmc2t_fit.py \
  --trainds mytrainds.csv \
  --modelout mymodel.pth \
  --hlayers 200 300 200 \
  --hactivation 'Sigmoid()' 'Tanh()' 'Sigmoid()' \
  --epochs 250

$ python pmc2t_fit_twin.py \
  --trainds mytrainds.csv \
  --modelout mymodeltwin.pth \
  --hlayers 200 300 200 \
  --hactivation 'Sigmoid()' 'Tanh()' 'Sigmoid()' \
  --epochs 250

$ python pmc2t_fit.py \
      --trainds mytrainds.csv \
      --modelout mymodel.pth \
      --hlayers 200 200 200 \
      --hactivation 'Sigmoid()' 'Sigmoid()' 'Sigmoid()' \
      --epochs 250 \
      --optimizer 'Adamax(lr=1.1e-2)'

$ python pmc2t_fit_twin.py \
    --trainds mytraintds.csv \
    --modelout mymodeltwin.pth \
    --hlayers 200 200 200 \
    --hactivation 'Sigmoid()' 'Sigmoid()' 'Sigmoid()' \
    --epochs 250 \
    --optimizer 'Adamax(lr=1.1e-2)'
```


## pmc2t_predict.py and pmc2t_predict_twin.py<a name="pmc2t_predict"/>
To get the usage of [pmc2t_predict.py](./pmc2t_predict.py) please run:
```bash
$ python pmc2t_predict.py --help
```

and you get:
```
usage: pmc2t_predict.py [-h]
                        --model MODEL_PATH
                        --ds DATASET_FILENAME
                        --predictionout PREDICTION_DATA_FILENAME
                        [--device DEVICE]

pmc2t_predict.py makes prediction of couples of coordinates of a parametric curve
on plane modeled with a pretrained multilayer perceptron with two output neurons

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL_PATH    model file
  --ds DATASET_FILENAME dataset file (csv format); only t-values are used
  --predictionout PREDICTION_DATA_FILENAME prediction data file (csv format)
  --device DEVICE       target device
```

To get the usage of [pmc2t_predict_twin.py](./pmc2t_predict_twin.py) please run:
```bash
$ python pmc2t_predict_twin.py --help
```

you will get the identical set of command line parameters of **pmc2t_predict.py**

Namely:
- **-h or --help** shows the above usage
- **--model** is the pth file of a model generated by **pmc2t_fit.py** (see **--modelout** command line parameter of **pmc2t_fit.py**). This argument is mandatory.
- **--ds** is the input dataset in csv format (no header at first line): program uses only the x values (first column). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **pmc2t_gen.py**. This argument is mandatory.
- **--predictionout** is the file name of prediction values. The content of this file is csv (no header at first line) and each line contains a triple of real numbers: the t value comes from input dataset and the prediction is the couple of values of x(t) and y(t) computed by multilayer perceptron model on t value; this argument is mandatory.
- **--device** is the target CUDA device where to perform math computations; default is **cpu**; to use default GPU pass **cuda**; please see [PyTorch CUDA semantic reference](https://pytorch.org/docs/stable/notes/cuda.html) for details.

### Example of pmc2t_predict.py and pmc2t_predict_twin.py usage
```bash
$ python pmc2t_predict.py --model mymodel.pth --ds mytestds.csv --predictionout myprediction.csv

$ python pmc2t_predict_twin.py --model mymodeltwin.pth --ds mytestds.csv --predictionout mypredictiontwin.csv
```


## pmc2t_plot.py<a name="pmc2t_plot"/>
To get the usage of [pmc2t_plot.py](./pmc2t_plot.py) please run:
```bash
$ python pmc2t_plot.py --help
```

and you get:
```
usage: pmc2t_plot.py [-h]
                     --ds DATASET_FILENAME
                     --prediction PREDICTION_DATA_FILENAME
                     [--savefig SAVE_FIGURE_FILENAME]

pmc2t_plot.py shows two overlapped x/y scatter graphs: the blue one is the
dataset, the red one is the prediction

optional arguments:
  -h, --help                     show this help message and exit
  --ds DATASET_FILENAME          dataset file (csv format)
  --prediction PREDICTION_DATA_FILENAME prediction data file (csv format)
  --savefig SAVE_FIGURE_FILENAME if present, the chart is saved on a file instead to be shown on screen
```

Namely:
- **-h or --help** shows the above usage
- **--ds** is an input dataset in csv format (no header at first line). Usually this parameter is the test dataset file passed to **pmc2t_predict.py**, but you could pass the training dataset passed to **pmc2t_fit.py**. This argument is mandatory.
- **--prediction** is the file name of prediction values generated by **pmc2t_predict.py** (see **--predictionout** command line parameter of **pmc2t_predict.py**). This argument is mandatory.
- **--savefig** if this argument is missing, the chart is shown on screen, otherwise this argument is the png output filename where **pmc2t_plot.py** saves the chart.

### Examples of pmc2t_plot.py usage
```bash
$ python pmc2t_plot.py --ds mytestds.csv --prediction myprediction.csv

$ python pmc2t_plot.py --ds mytrainds.csv --prediction myprediction.csv --savefig mychart.png
```
