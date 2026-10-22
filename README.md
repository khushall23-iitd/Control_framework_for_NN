# A-Tutorial-on-Control-Theoretic-Framework-for-Neural-Network-Training-Dynamics
This folder contains the MATLAB scripts used to produce the experimental
results reported in *"A Tutorial on Control-Theoretic Framework for Neural
Network Training Dynamics"*. The paper interprets neural network training as
a closed-loop feedback control system: the parameter vector is the state,
the gradient is the measured feedback signal, the optimizer is the
controller, and the learning rate is a feedback gain. A Control Lyapunov
Function (CLF)-motivated adaptive learning rate is proposed, which selects
the gain from the current loss and gradient, and is compared against
standard SGD, SGD with momentum, and Adam.

Two experiments are shown: XOR on a 2–2–1 network, and CIFAR-10 on a
five-layer CNN.

## Files

- `XOR.m` - Binary XOR classification with a 2-2-1 feedforward network
  (2 inputs, 2 hidden neurons, 1 sigmoid output), trained with full-batch
  binary cross-entropy loss for 1000 epochs. Compares SGD, SGD with
  momentum, Adam, and the proposed CLF-based gain. Produces training-loss,
  accuracy, adaptive step-size, and decision-boundary plots.
- `CIFAR10.m` - CIFAR-10 image classification with a five-layer CNN
  (three convolution + batch-norm + ReLU blocks, followed by a fully
  connected layer and a softmax output), trained with mini-batch
  (stochastic) gradients for 50 epochs. Compares the same four optimizers
  and reports test accuracy and training-loss curves. Downloads the
  CIFAR-10 dataset automatically on first run.

## Requirements

- MATLAB R2022b (or later) with the Deep Learning Toolbox (required by
  `CIFAR10.m` for `dlnetwork`, `dlarray`, and `dlfeval`).
- Internet access on first run of `CIFAR10.m` to download the CIFAR-10
  dataset from `https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz`.

## Reproducibility

Both scripts fix the random seed (`rng(10)`) and initialize all compared
optimizers from the same starting parameters, so that differences in the
reported curves are attributable to the update rule rather than
initialization. Exact learning rates, momentum/Adam coefficients, and CLF
design parameters ($\alpha$, $\epsilon$) used for the paper's figures are
documented in the "Experimental Records and Availability" section of the
paper.

