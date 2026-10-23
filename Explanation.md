# `CIFAR10.m`

### 2.1 Problem and model

* Task: 10-class image classification on CIFAR-10.
* Data: 50,000 training images and 10,000 test images, each $32\times32\times3$.
* Gradients are **mini-batch stochastic gradients** (batch size 128). This differs from XOR, where gradients are exact. The deterministic descent guarantee of the paper therefore does not apply directly. The experiment is empirical.
* Loss: softmax cross-entropy.

Network ("five-layer CNN": 3 convolutional + 2 fully connected layers):

| # | Layer | Name | Output size | Learnable parameters |
|---|---|---|---|---|
| 0 | Image input, no normalization | `input` | 32×32×3 | – |
| 1 | Conv 3×3, 32 filters, `same` padding | `conv1` | 32×32×32 | 896 |
| | BatchNorm | `bn1` | 32×32×32 | 64 |
| | ReLU | `relu1` | 32×32×32 | – |
| | MaxPool 2×2, stride 2 | `pool1` | 16×16×32 | – |
| 2 | Conv 3×3, 64 filters, `same` | `conv2` | 16×16×64 | 18,496 |
| | BatchNorm | `bn2` | 16×16×64 | 128 |
| | ReLU | `relu2` | 16×16×64 | – |
| | MaxPool 2×2, stride 2 | `pool2` | 8×8×64 | – |
| 3 | Conv 3×3, 128 filters, `same` | `conv3` | 8×8×128 | 73,856 |
| | BatchNorm | `bn3` | 8×8×128 | 256 |
| | ReLU | `relu3` | 8×8×128 | – |
| 4 | Fully connected, 128 units (input 8192) | `fc1` | 128 | 1,048,704 |
| | ReLU | `relu4` | 128 | – |
| 5 | Fully connected, 10 units | `fc_out` | 10 | 1,290 |

Total: 1,143,690 learnable parameters (16 learnable arrays). The softmax is applied inside `modelGradients` and `evaluateAccuracy`, not as a network layer. MATLAB default initializers apply (Glorot weights, zero biases, BN scale 1 / offset 0), drawn from the global generator seeded by `rng(10)`.

### 2.2 All settings needed for reproduction

| Quantity | Variable | Value |
|---|---|---|
| MATLAB version | – | R2022b (Deep Learning Toolbox required) |
| Random seed | `rng(10)` | 10 (first executable statement after `clc; clear; close all;`) |
| Dataset URL | `cifarURL` | `https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz` |
| Data folder | `dataFolder` | `fullfile(pwd,'cifar10_data')` |
| Training / test images | – | 50,000 / 10,000 |
| Scaling | – | `single(X)/255` |
| Normalization | `mu`, `sig` | per-channel mean and std of the training set (std with $+10^{-8}$) |
| Epochs | `numEpochs` | 50 |
| Mini-batch size | `miniBatchSize` | 128 |
| Iterations per epoch | `numIterationsPerEpoch` | `floor(50000/128)` = 390 |
| Images used per epoch | – | 390·128 = 49,920 (80 leftover images dropped each epoch, different ones each epoch) |
| SGD learning rate | `lr_sgd` | 0.01 |
| Momentum learning rate | `lr_mom` | 0.01 |
| Momentum coefficient | `momBeta` | 0.9 |
| Adam learning rate | `lr_adam` | 0.001 |
| Adam $\beta_1$ | `beta1` | 0.9 |
| Adam $\beta_2$ | `beta2` | 0.999 |
| Adam $\varepsilon$ | `epsAdam` | $10^{-8}$ |
| CLF $\alpha$ | `alpha_clf` | 1.0 |
| CLF $\epsilon$ | hard-coded in the gain expression | $10^{-12}$ |
| Loss numerical term | `modelGradients` | $10^{-8}$ added to probabilities inside `log` |
| CLF cap $\bar\eta$ | – | none (uncapped) |
| Momentum / Adam state init | `velocityMOM`, `avgGradADAM`, `avgSqADAM` | zeros (same shapes as learnables) |
| Execution device | – | CPU (no `gpuArray` is used) |

Reproduction notes:

* The network is built once with `createCIFAR10CNN` right after the data are loaded. No random call occurs between `rng(10)` and the network construction, so the initial weights are determined by the seed. The same `baseNet` is copied to all four methods.
* `idx = randperm(numTrain)` is drawn **once per epoch** and shared by all four methods, so all methods see identical mini-batch sequences. Because `randperm` consumes the global generator after the network was built, the order depends on the seed and on the full call sequence of the script.
* Results can still differ slightly across MATLAB versions, toolbox versions, and hardware.

### 2.3 Line-by-line walkthrough

**Header and seed**

```matlab
clc; clear; close all;
rng(10);
```

**Dataset download and loading**

```matlab
dataFolder = fullfile(pwd,'cifar10_data');
if ~exist(dataFolder,'dir'), mkdir(dataFolder); end
...
if ~exist(fullfile(dataFolder,'cifar-10-batches-mat'),'dir')
    websave(zipFile,cifarURL);
    untar(zipFile,dataFolder);
end
[XTrain,YTrain,XTest,YTest,classes] = loadCIFAR10(dataFolder);
```

Creates `cifar10_data` in the current folder, downloads and extracts the MATLAB-format archive only if it is not already extracted, then loads it with `loadCIFAR10` (Sec. 2.4).

**Normalization**

```matlab
XTrain = single(XTrain)/255;
XTest  = single(XTest)/255;
mu  = mean(XTrain,[1 2 4]);
sig = std(XTrain,0,[1 2 4]) + 1e-8;
XTrain = (XTrain - mu)./sig;
XTest  = (XTest  - mu)./sig;
```

Pixels are scaled to $[0,1]$, then standardized per colour channel. `mu` and `sig` are $1\times1\times3$ and are computed from the training set only, then applied to both sets (no test-set leakage). The network's input layer has `'Normalization','none'` so it performs no further normalization.

**Hyper-parameters, iteration count**

```matlab
numEpochs = 50; miniBatchSize = 128;
lr_sgd = 0.01;
lr_mom = 0.01; momBeta = 0.9;
lr_adam = 0.001; beta1 = 0.9; beta2 = 0.999; epsAdam = 1e-8;
alpha_clf = 1.0;
numTrain = size(XTrain,4);
numIterationsPerEpoch = floor(numTrain/miniBatchSize);   % 390
```

**Identical initial networks and optimizer states**

```matlab
baseNet = createCIFAR10CNN(classes);
netSGD = baseNet; netMOM = baseNet; netADAM = baseNet; netCLF = baseNet;
velocityMOM = initializeLike(netMOM.Learnables);
avgGradADAM = initializeLike(netADAM.Learnables);
avgSqADAM   = initializeLike(netADAM.Learnables);
```

Four copies of the same initial `dlnetwork`. `initializeLike` produces zero tables with the same layout as `Learnables` for the momentum velocity and the two Adam moments.

**Storage**

`lossSGD/MOM/ADAM/CLF` (average training loss per epoch), `accSGD/MOM/ADAM/CLF` (test accuracy per epoch), `lambdaCLF` (average CLF gain per epoch), and `iterationADAM` (global Adam step counter, not reset between epochs).

**Training loop**

For each epoch:

1. `idx = randperm(numTrain)` shuffles the training set. Epoch accumulators are reset.
2. For each of the 390 iterations: the batch indices are `idx((it-1)*128+1 : it*128)`; `XBatch`/`YBatch` are extracted; `dlX = dlarray(XBatch,'SSCB')` (spatial, spatial, channel, batch).
3. Each method then evaluates its own gradient at its own parameters with `dlfeval(@modelGradients, net, dlX, YBatch, classes)`:

   * **SGD**
     ```matlab
     netSGD.Learnables = dlupdate(@(p,g) p - lr_sgd*g, netSGD.Learnables, gradients);
     ```
   * **SGD-Momentum**
     ```matlab
     velocityMOM = dlupdate(@(v,g) momBeta*v + g, velocityMOM, gradients);
     netMOM.Learnables = dlupdate(@(p,v) p - lr_mom*v, netMOM.Learnables, velocityMOM);
     ```
     Same heavy-ball form as in `XOR.m`.
   * **Adam**: `iterationADAM` is incremented, then `adamUpdateCustom` (Sec. 2.4) updates parameters and moments.
   * **CLF-GD**
     ```matlab
     lossVal   = double(extractdata(loss));
     gradNorm2 = gradientNormSquared(gradients);
     lambda    = lossVal/(gradNorm2 + alpha_clf*lossVal + 1e-12);
     netCLF.Learnables = dlupdate(@(p,g) p - lambda*g, netCLF.Learnables, gradients);
     ```
     The gain is computed **per mini-batch** from that batch's loss and the squared 2-norm of the full gradient (all layers together, a single scalar $\lambda$ for the whole network). It is applied uncapped.

   Each iteration's loss is added to the epoch accumulator (`epochLoss*`), and for CLF the gain is also accumulated (`epochLambda`).

4. At the end of the epoch:
   * `loss*(epoch)` = accumulated loss / 390. This is the average of the mini-batch losses **seen during the epoch while parameters were changing**. It is not a re-evaluation of the full training set after the epoch.
   * `lambdaCLF(epoch)` = mean CLF gain over the 390 iterations.
   * `acc*(epoch)` = test accuracy of each network via `evaluateAccuracy` on the full 10,000-image test set.
   * One line is printed: `Epoch ee/50 | SGD .. | MOM .. | Adam .. | CLF ..`.

**Final table**

Prints the **peak** (maximum over epochs) test accuracy of each method, not the last-epoch accuracy.

**Plots and files written to the current folder**

| Figure | Content | Saved as (600 dpi) |
|---|---|---|
| 1 | Test accuracy vs. epoch | `CIFAR10_Actual_Test_Accuracy.png` |
| 2 | Training cross-entropy loss vs. epoch (data behind the CIFAR-10 training-loss figure in the paper) | `CIFAR10_Actual_Training_Loss.png` |
| 3 | Average CLF gain $\lambda_k$ vs. epoch | `CIFAR10_CLF_Adaptive_Gain.png` |

### 2.4 Local functions

**`createCIFAR10CNN(classes)`**: builds the layer array of Sec. 2.1 and returns `dlnetwork(layerGraph(layers))`. `numClasses = numel(classes)` = 10.

**`modelGradients(net, dlX, Y, classes)`**

```matlab
dlY  = forward(net,dlX);
P    = softmax(dlY);
T    = dlarray(oneHotManual(Y,classes),'CB');
loss = -mean(sum(T.*log(P + 1e-8),1));
gradients = dlgradient(loss, net.Learnables);
```

Computes the mean cross-entropy over the batch, $\mathcal J=-\frac1B\sum_b\sum_c T_{cb}\log(P_{cb}+10^{-8})$, and its gradient w.r.t. all learnables via automatic differentiation.

**`oneHotManual(Y, classes)`**: converts the categorical batch labels into a $10\times B$ single-precision one-hot matrix by matching label strings against `classes`.

**`evaluateAccuracy(net, XTest, YTest, classes, miniBatchSize)`**: loops over the test set in batches of 128 (79 batches, the last has 16 images), takes `forward`, `softmax`, then `max` over classes, compares predictions to `YTest`, and returns the percentage correct.

**`gradientNormSquared(gradients)`**: returns $\sum_i \lVert g_i\rVert_2^2$ over all 16 learnable arrays, i.e. $\lVert\nabla\mathcal J\rVert^2$ of the full parameter vector.

**`initializeLike(T)`**: copy of a `Learnables` table with every value replaced by zeros of the same size and type.

**`adamUpdateCustom(params, gradients, avgGrad, avgSqGrad, iteration, lr, beta1, beta2, epsAdam)`**: for every learnable array

$$m\leftarrow\beta_1m+(1-\beta_1)g,\quad v\leftarrow\beta_2v+(1-\beta_2)g^2,$$
$$\hat m=\frac{m}{1-\beta_1^{t}},\quad\hat v=\frac{v}{1-\beta_2^{t}},\quad p\leftarrow p-\mathrm{lr}\,\frac{\hat m}{\sqrt{\hat v}+\varepsilon_{\rm Adam}}$$

with $t=$ `iteration` (global step, 1 to 19,500).

**`loadCIFAR10(dataFolder)`**: reads `data_batch_1..5.mat` and `test_batch.mat`. Each row of `data` is a 3072-vector stored channel-by-channel. `reshape(data',32,32,3,[])` followed by `permute([2 1 3 4])` yields images in height×width×channel×N order. Labels (0–9) are converted to categorical using `batches.meta.mat` class names (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).

### 2.5 Behaviours of the code worth knowing

* **BatchNorm is always in "training" mode.** `modelGradients` and `evaluateAccuracy` both call `forward(net,dlX)` and discard the returned state. Hence the network's running mean/variance are never updated, and test-time accuracy is computed using the statistics of each test mini-batch (128 images), not population statistics. This applies equally to all four methods. If you want to reproduce the reported numbers, do not change this.
* **Training loss is a running average** over the epoch (see 2.3), whereas **accuracy is on the test set**. The paper's CIFAR-10 figure shows training loss.
* **Reported peak accuracy** is `max(acc*)` over the 50 epochs.
* The shared mini-batch order isolates the effect of the update rule from the data order.

### 2.6 How to run

1. MATLAB R2022b or later with the Deep Learning Toolbox; internet access on the first run.
2. Open the `matlab_code` folder in MATLAB and run `CIFAR10`.
3. The first run downloads about 163 MB, extracts it into `./cifar10_data`, and trains 4 networks × 50 epochs × 390 iterations. Progress is printed once per epoch.
4. Outputs: console log, peak-accuracy table, and the three PNGs listed above, saved in the current folder.

---

## Part 3 — Side-by-side summary of the two experiments

| Item | XOR | CIFAR-10 |
|---|---|---|
| Model | 2-2-1 sigmoid FFNN, 9 parameters | 3 conv + 2 FC CNN with BN, 1,143,690 parameters |
| Loss | Binary cross-entropy | Softmax cross-entropy ($+10^{-8}$ inside log) |
| Data | 4 samples | 50,000 train / 10,000 test |
| Gradient | Full batch (exact) | Mini-batch of 128 (stochastic) |
| Updates | 1000 | 50 × 390 = 19,500 |
| SGD $\eta$ | 0.8 | 0.01 |
| Momentum $\eta,\beta$ | 0.8, 0.9 | 0.01, 0.9 |
| Adam $\eta,\beta_1,\beta_2,\varepsilon$ | 0.05, 0.9, 0.999, $10^{-8}$ | 0.001, 0.9, 0.999, $10^{-8}$ |
| CLF $\alpha$ | 0.4 | 1.0 |
| CLF $\epsilon$ | $10^{-12}$ | $10^{-12}$ |
| CLF cap $\bar\eta$ | none | none |
| Implicit gain bound $1/\alpha$ | 2.5 | 1.0 |
| Seed | `rng(10)` | `rng(10)` |
| Metrics logged | loss, training accuracy, $\lambda_k$ | training loss, test accuracy, mean $\lambda_k$ |
| Toolboxes | none | Deep Learning Toolbox |
