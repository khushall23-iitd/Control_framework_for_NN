# Explanation of `XOR.m` and `CIFAR10.m`

This document walks through both MATLAB scripts from the first line to the last. It also lists every numerical setting needed to reproduce the reported experiments.

Both scripts compare four update rules applied to the same network from the same starting point:

| Name in code | Update rule |
|---|---|
| SGD | $\theta_{k+1}=\theta_k-\eta\,g_k$ |
| SGD-Momentum | $v_{k+1}=\beta v_k+g_k,\quad \theta_{k+1}=\theta_k-\eta\,v_{k+1}$ |
| Adam | bias-corrected first/second moment, coordinate-wise step |
| CLF-GD (proposed) | $\theta_{k+1}=\theta_k-\lambda_k g_k,\quad \lambda_k=\dfrac{\mathcal J(\theta_k)}{\lVert g_k\rVert^2+\alpha\,\mathcal J(\theta_k)+\epsilon}$ |

Here $g_k=\nabla\mathcal J(\theta_k)$. In the control interpretation of the paper, $\theta$ is the plant state, $g_k$ is the measured feedback, and the step size ($\eta$ or $\lambda_k$) is the feedback gain.

**Important:** in both scripts the CLF gain is used **uncapped**, i.e. without the upper bound $\bar\eta$ of Eq. (bounded CLF gain) in the paper. No backtracking or safety filter is applied. The gain is still bounded implicitly, because

$$\lambda_k=\frac{\mathcal J}{\lVert g\rVert^2+\alpha\mathcal J+\epsilon}<\frac{1}{\alpha}.$$

So $\lambda_k<1/0.4=2.5$ for XOR and $\lambda_k<1/1.0=1$ for CIFAR-10. Neither bound is a proof of loss descent, because that would require $\lambda_k<2/L_{\mathcal J}$.

---

## Part 1 — `XOR.m`

### 1.1 Problem and model

* Task: learn XOR with a $2$-$2$-$1$ feed-forward network (2 inputs, 2 hidden neurons, 1 output).
* Hidden activation: sigmoid. Output activation: sigmoid.
* Loss: mean binary cross-entropy over the 4 samples.
* Training is **full-batch**: all 4 samples are used at every iteration, so the gradient is exact (deterministic), matching the assumptions of the Lyapunov analysis.
* One "epoch" equals exactly one parameter update.

Forward model for sample $i$:

$$z_{1i}=w_{11}x_{i1}+w_{12}x_{i2}+b_1,\qquad z_{2i}=w_{21}x_{i1}+w_{22}x_{i2}+b_2$$
$$h_{ji}=\sigma(z_{ji}),\qquad s_i=v_1h_{1i}+v_2h_{2i}+b_3,\qquad \hat y_i=\sigma(s_i),\quad \sigma(x)=\frac1{1+e^{-x}}$$

Loss:

$$\mathcal J(\theta)=-\frac1N\sum_{i=1}^{N}\big[y_i\log\hat y_i+(1-y_i)\log(1-\hat y_i)\big],\quad N=4$$

### 1.2 Parameter vector

$$\theta=[w_{11}\ w_{12}\ w_{21}\ w_{22}\ b_1\ b_2\ v_1\ v_2\ b_3]^\top\in\mathbb R^9$$

| Index | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Symbol | `w11` | `w12` | `w21` | `w22` | `b1` | `b2` | `v1` | `v2` | `b3` |

### 1.3 All settings needed for reproduction

| Quantity | Variable | Value |
|---|---|---|
| MATLAB version | – | R2022b |
| Random seed | `rng(10)` | 10 |
| Inputs | `X` | `[0 0; 0 1; 1 0; 1 1]` |
| Targets | `Y` | `[0; 1; 1; 0]` |
| Samples | `N` | 4 (full batch) |
| Epochs (= updates) | `epochs` | 1000 |
| Initial parameters | `theta0` | `0.5*randn(9,1)` drawn right after `rng(10)` |
| SGD learning rate | `eta_sgd` | 0.8 |
| Momentum learning rate | `eta_mom` | 0.8 |
| Momentum coefficient | `beta_mom` | 0.9 |
| Adam learning rate | `eta_adam` | 0.05 |
| Adam $\beta_1$ | `beta1` | 0.9 |
| Adam $\beta_2$ | `beta2` | 0.999 |
| Adam $\varepsilon$ | `eps_adam` | $10^{-8}$ |
| CLF $\alpha$ | `alpha_clf` | 0.4 |
| CLF $\epsilon$ | hard-coded in the gain expression | $10^{-12}$ |
| Output clipping | inside `lossGrad_XOR_221` | $\hat y\in[10^{-10},\,1-10^{-10}]$ |
| Classification threshold | `classification_accuracy` | $\hat y\ge0.5$ |
| Momentum / Adam state init | `vel_mom`, `m_adam`, `v_adam` | zeros(9,1) |
| CLF cap $\bar\eta$ | – | none (uncapped) |

Reproduction notes:

* `rng(10)` must be called before `randn`, and no other random call may occur between them. In the script this is the case.
* `theta0` is generated once and copied to all four optimizers, so every method starts at the identical point.

### 1.4 Line-by-line walkthrough

**Header and seed**

```matlab
clc; clear; close all;
rng(10);
```

Clears the console, workspace and figures, then fixes the random generator so that `theta0` is identical on every run.

**Dataset**

```matlab
X = [0 0; 0 1; 1 0; 1 1];
Y = [0; 1; 1; 0];
N = size(X,1);
```

The four XOR input pairs (rows of `X`) and their labels. `N = 4`.

**Hyper-parameters**

```matlab
epochs = 1000;
eta_sgd = 0.8;
eta_mom = 0.8;  beta_mom = 0.9;
eta_adam = 0.05; beta1 = 0.9; beta2 = 0.999; eps_adam = 1e-8;
alpha_clf = 0.4;
```

These are the values in the table above.

**Initialization**

```matlab
theta0 = 0.5 * randn(9,1);
theta_sgd = theta0; theta_mom = theta0; theta_adam = theta0; theta_clf = theta0;
vel_mom = zeros(9,1);
m_adam = zeros(9,1);  v_adam = zeros(9,1);
```

Each optimizer owns a private copy of the parameters. Momentum has one auxiliary state (the velocity). Adam has two (the first and second moment estimates).

**Storage**

```matlab
loss_*  = zeros(epochs,1);  acc_* = zeros(epochs,1);  lambda_clf_store = zeros(epochs,1);
```

Pre-allocated logs of loss, accuracy and the CLF gain for each iteration.

**Training loop** (`for k = 1:epochs`)

Each of the four blocks calls `lossGrad_XOR_221` to get the loss `L`, gradient `g`, and predictions `yhat` at the current parameters, then updates.

1. **SGD**
   ```matlab
   theta_sgd = theta_sgd - eta_sgd * g;
   ```
   Fixed-gain proportional feedback.
2. **SGD + Momentum**
   ```matlab
   vel_mom   = beta_mom * vel_mom + g;
   theta_mom = theta_mom - eta_mom * vel_mom;
   ```
   Heavy-ball form. The gradient is **not** scaled by $(1-\beta)$, so the effective gain in steady state is $\eta/(1-\beta)=8$.
3. **Adam**
   ```matlab
   m_adam = beta1*m_adam + (1-beta1)*g;
   v_adam = beta2*v_adam + (1-beta2)*(g.^2);
   m_hat  = m_adam/(1-beta1^k);
   v_hat  = v_adam/(1-beta2^k);
   theta_adam = theta_adam - eta_adam * m_hat ./ (sqrt(v_hat)+eps_adam);
   ```
   Standard Adam with bias correction. The loop index `k` is the Adam time step.
4. **CLF-GD**
   ```matlab
   lambda_clf = L / (norm(g)^2 + alpha_clf*L + 1e-12);
   theta_clf  = theta_clf - lambda_clf * g;
   ```
   The gain is computed from the current loss and the squared gradient 2-norm. It is a scalar, so the step is along $-g$ (a scalar gain, not a diagonal one as in Adam). `lambda_clf` is stored in `lambda_clf_store(k)`.

Logging detail: for iteration `k`, `loss_*(k)` and `acc_*(k)` are evaluated at the parameters **before** the k-th update. The logged curve therefore starts at the loss of `theta0`.

**Final predictions**

`print_predictions` is called for each optimizer. It re-evaluates the loss and predictions at the parameters after all 1000 updates and prints a table (`x1 x2 y yhat pred`) together with the final loss.

**Plots**

| Figure | Content |
|---|---|
| 1 | Training BCE loss vs. epoch for the four methods (this is the data behind the XOR loss figure) |
| 2 | Classification accuracy (%) vs. epoch |
| 3 | CLF step size $\lambda_k$ vs. epoch |
| 4 | Decision boundary of CLF-GD (contour at $\hat y=0.5$ over $[-0.2,1.2]^2$ on a $300\times300$ grid) |

XOR.m does not save the figures; export them manually if needed.

### 1.5 Local functions

**`lossGrad_XOR_221(theta, X, Y)`**: forward pass, loss, and hand-written backpropagation.

* Unpacks `theta` into the nine named parameters.
* Forward pass as in Sec. 1.1, vectorized over the 4 samples.
* Clips `yhat` into $[10^{-10},1-10^{-10}]$ so that `log` is finite. The clip is not differentiated through; the backward pass uses the clipped `yhat` as if it were the true output.
* Backpropagation. With sigmoid output and BCE, the output error simplifies to
  $$\delta^o_i=\hat y_i-y_i,$$
  and the hidden errors are
  $$\delta^h_{ji}=\delta^o_i\,v_j\,h_{ji}(1-h_{ji}).$$
  The gradient components, each averaged over $N$, are
  $$\partial_{w_{j1}}\mathcal J=\tfrac1N\sum_i\delta^h_{ji}x_{i1},\quad
  \partial_{w_{j2}}\mathcal J=\tfrac1N\sum_i\delta^h_{ji}x_{i2},\quad
  \partial_{b_j}\mathcal J=\tfrac1N\sum_i\delta^h_{ji},$$
  $$\partial_{v_j}\mathcal J=\tfrac1N\sum_i\delta^o_ih_{ji},\qquad
  \partial_{b_3}\mathcal J=\tfrac1N\sum_i\delta^o_i.$$
* Returns the gradient stacked in the same order as `theta`.

**`sigmoid(x)`**: $1/(1+e^{-x})$.

**`classification_accuracy(yhat, Y)`**: thresholds at 0.5 and returns the percentage of correct labels.

**`print_predictions(name, theta, X, Y)`**: console report described above.

**`plot_decision_boundary(theta, X, Y)`**: evaluates `predict_XOR_221` on the grid, draws filled contours, the 0.5 level set as a thick black line, and the class 0 (circles) / class 1 (squares) points.

**`predict_XOR_221(theta, X)`**: forward pass only (no clipping, no loss), used for the grid.

### 1.6 How to run

1. Open MATLAB R2022b (or later) in the `matlab_code` folder.
2. Run `XOR` (no toolboxes required).
3. The console prints the final predictions of all four methods and four figures open.

To vary the CLF design parameter (the paper's figure compares different $\alpha$ values), change `alpha_clf` and re-run. The script as provided uses only $\alpha=0.4$.

---

## Part 2 — `CIFAR10.m`

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
