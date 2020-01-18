# Fitting with MLP using PyTorch
Goal of this repository is to share programs that fit some kinds of curves by high configurable multilayer perceptron (MLP) neural network written in Python 3 using PyTorch.

## One variable function fitting
The project [One variable function fitting](./one-variable-function-fitting) implements the fitting of a continuous and limited real-valued function defined in a closed interval of the reals.

## Parametric curve on plane fitting
The project [Parametric curve on plane fitting](./parametric-curve-on-plane-fitting) implements the fitting of a continuous and limited real-valued parametric curve on plane where parameter belongs to a closed interval of the reals. It implements two alternative techniques: the official one implements one MLP that fits a vector function f(t) = [x(t), y(t)] instead the 'twin' variant implements a pair of twins of MLPs that fit separately the one variable functions x(t) and y(t).

