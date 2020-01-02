# Fitting with MLP using-PyTorch
The goal of this repository is to share programs that fit some kinds of curves by high configurable MLP (Multi-Layer Perceptron) written in Python 3 using PyTorch.

## One variable function fitting
The project [One variable function fitting](./one-variable-function-fitting) implements the fitting of a continuous and limited real-valued function defined in a closed interval of the reals.

## Parametric curve on plane fitting
The project [Parametric curve on plane fitting](./parametric-curve-on-plane-fitting) implements the fitting of a continuous and limited real-valued parametric curve on plane where parameter belongs to a closed interval of the reals.

## Parametric curve on plane fitting (variant 2)
The project [Parametric curve on plane fitting (variant 2)](./parametric-curve-on-plane-fitting-vnt2) is like previous project: same purpose and same command line interface. The difference is in the implementation: the previous project use one model with 2 output neurons, this one uses two different linear models trained separately.
