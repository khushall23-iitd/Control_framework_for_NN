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
- `CIFAR10.m` - CIFAR-10 image classification with five learned layers
  (three convolutional layers and two fully connected layers), plus batch
  normalization, ReLU, and pooling. Trained with mini-batch gradients for
  50 epochs, it compares the same four optimizers and reports test accuracy
  and training-loss curves. Downloads the CIFAR-10 dataset automatically on
  first run.

### Explanation

#### `XOR.m`

1. **Prepare the experiment.** The script clears the MATLAB workspace and
  figures, then sets the random seed to 10 so the initial parameters are
  reproducible.
2. **Define the data and network.** The four possible two-bit inputs are
  paired with their XOR labels: the output is 1 when the inputs differ and
  0 when they match. The network has two input values, two sigmoid hidden
  neurons, and one sigmoid output. Its nine parameters are four input-to-
  hidden weights, two hidden biases, two hidden-to-output weights, and one
  output bias.
3. **Set training options and initialize models.** Four optimizer settings
  are defined: SGD, SGD with momentum, Adam, and CLF-based gradient descent
  (CLF-GD). A single random parameter vector is copied to all four models,
  making the optimizer comparison start from the same network. Momentum and
  Adam also receive zero-initialized state variables.
4. **Compute predictions, loss, and gradients.** For each optimizer,
  `lossGrad_XOR_221` runs the forward pass: it calculates the two hidden
  activations, combines them at the output neuron, and applies the sigmoid.
  Predictions are clipped slightly away from 0 and 1 for numerical
  stability. The function computes the average binary cross-entropy loss
  and uses backpropagation to return the gradient for all nine parameters.
5. **Update the parameters for each epoch.** Each optimizer uses the
  gradients to update its own copy of the network. SGD takes a fixed-size
  step; momentum accumulates a running gradient; Adam uses bias-corrected
  first and second gradient moments; and CLF-GD sets its step size to
  `loss / (squared gradient norm + alpha * loss + 1e-12)`. The script stores
  the loss, threshold-based classification accuracy, and CLF step size at
  each epoch.
6. **Report and plot results.** The final predictions are printed for each
  input and optimizer. Separate figures compare training loss and accuracy,
  show the CLF step size over time, and draw the CLF-GD decision boundary
  over the four XOR examples. Helper functions provide sigmoid activation,
  accuracy calculation, prediction output, and decision-boundary plotting.

#### `CIFAR10.m`

1. **Load the dataset.** The script creates a local `cifar10_data` folder.
  If the CIFAR-10 MATLAB files are not present, it downloads and extracts
  them. `loadCIFAR10` reads the five training batches and the test batch,
  reshapes the flattened pixel arrays into 32-by-32 RGB images, and converts
  the numeric labels into categorical class names.
2. **Normalize the images.** Pixel values are converted to single precision
  and scaled from 0-255 to 0-1. The per-color-channel mean and standard
  deviation are calculated from the training images; those same values are
  used to standardize both the training and test images.
3. **Configure training.** The run uses 50 epochs and batches of 128 images.
  It sets the learning rates and other parameters for SGD, momentum, Adam,
  and CLF-GD, then calculates the number of complete training batches in an
  epoch. A final incomplete training batch is not included.
4. **Build and copy the CNN.** `createCIFAR10CNN` creates one network, then
  copies it to four model variables so all optimizers begin with the same
  initial network. The architecture has five learned layers (three
  convolutional layers and two fully connected layers), arranged as:

  | Stage | Layer | Output size |
  | --- | --- | --- |
  | Input | RGB image | 32 x 32 x 3 |
  | 1 | 3 x 3 convolution, 32 filters; batch normalization; ReLU; 2 x 2 max pooling | 16 x 16 x 32 |
  | 2 | 3 x 3 convolution, 64 filters; batch normalization; ReLU; 2 x 2 max pooling | 8 x 8 x 64 |
  | 3 | 3 x 3 convolution, 128 filters; batch normalization; ReLU | 8 x 8 x 128 |
  | 4 | Fully connected, 128 units; ReLU | 128 |
  | 5 | Fully connected, one score per class | 10 |

  The convolutions use same padding. The last layer produces class scores;
  `modelGradients` applies softmax to turn them into class probabilities.
  Thus, softmax is part of the loss calculation rather than a separate
  layer in `createCIFAR10CNN`.

5. **Initialize optimizer state and result storage.** The script prepares
  separate momentum and Adam accumulators, plus arrays for each optimizer's
  epoch losses, test accuracies, and the average CLF gain.

6. **Train on shuffled mini-batches.** At the start of each epoch, training
  images are shuffled and split into batches. For each batch,
  `modelGradients` runs the network, creates one-hot class targets, computes
  softmax cross-entropy, and uses automatic differentiation to obtain
  gradients. The four models then update their parameters independently:
  SGD uses the raw gradient, momentum uses its running gradient, Adam uses
  bias-corrected gradient moments, and CLF-GD computes an adaptive gain
  from the loss and squared gradient norm. Batch losses and CLF gains are
  accumulated for the epoch.

7. **Measure and report performance.** After each epoch, the script stores
  average training loss and evaluates all four models on the complete test
  set in batches. `evaluateAccuracy` selects the highest-probability class
  and reports the percentage of correct predictions. The epoch's test
  accuracies are printed; after training, the peak accuracy for each
  optimizer is displayed.
  
8. **Plot the training results.** The script creates and saves three
  600-DPI figures: test accuracy, training cross-entropy loss, and the
  average CLF adaptive gain across epochs.

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

