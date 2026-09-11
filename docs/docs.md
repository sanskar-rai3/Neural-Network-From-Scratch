# Neural Network From Scratch in C {#mainpage}

A feed-forward neural network implemented entirely in C, without machine-learning frameworks.

The purpose of this project is to understand how neural networks work internally by implementing the underlying mathematics and components directly rather than hiding them behind high-level libraries.

The network currently supports matrix operations, EMNIST loading, dense layers, activation functions, loss functions, backpropagation, configurable multi-layer networks, SGD optimization, EMNIST training, inference, accuracy evaluation, and sample rendering.

## Overview

The basic data flow through the network is:

```
EMNIST image
    |
    v
Input matrix
    |
    v
Dense layer
    |
    v
Activation
    |
    v
Dense layer
    |
    v
Softmax
    |
    v
Class probabilities
    |
    v
Loss
    |
    v
Backpropagation
    |
    v
Parameter update
```

## Matrix Library

The neural network is built on top of a custom matrix implementation.

Matrices use row-major storage.

For a matrix with rows and cols, the element at `(row, col)` is stored at:

```c
mat->data[row * mat->cols + col]
```

The matrix library provides operations including:

- Matrix creation and destruction
- Element access
- Matrix copying
- Matrix addition and subtraction
- In-place arithmetic
- Scalar multiplication
- Hadamard product
- Matrix multiplication
- Transposition
- Element-wise operations
- Random initialization
- Matrix filling

### Matrix Multiplication

For two matrices:

```
A = m x n
B = n x p
```

the result is:

```
C = m x p
```

Each element is calculated as:

```
C[i][j] = sum(A[i][k] * B[k][j])
```

Matrix multiplication is the primary operation used by dense layers.

## Dense Layers

A dense layer performs an affine transformation:

```
Z = XW + b
```

where:

- `X` = input
- `W` = weights
- `b` = bias
- `Z` = output before activation

For example:

| | Shape |
|---|---|
| Input | 64 x 784 |
| Weights | 784 x 128 |
| Bias | 1 x 128 |
| Output | 64 x 128 |

The same operation therefore processes an entire batch of samples at once.

Weights and biases are initialized when the layer is created.

During training, the layer also stores the information required for backpropagation.

## Activation Functions

The network supports several activation functions:

- ReLU
- Leaky ReLU
- Tanh
- Sigmoid
- Softmax
- Linear / None

Activations are applied to the output of dense layers.

### ReLU

ReLU is defined as:

```
f(x) = max(0, x)
```

It is primarily used in hidden layers.

### Leaky ReLU

Leaky ReLU is defined as:

```
f(x) = 0.01x    if x < 0
       x        if x >= 0
```

It allows a small gradient to pass through negative values.

### Sigmoid

Sigmoid is defined as:

```
sigma(x) = 1 / (1 + exp(-x))
```

Its output is in the range:

```
0 < sigma(x) < 1
```

### Tanh

Tanh is defined as:

```
tanh(x)
```

Its output is in the range:

```
-1 < tanh(x) < 1
```

### Softmax

Softmax converts a vector of logits into a probability distribution.

For a vector `z`:

```
softmax(z_i) = exp(z_i) / sum(exp(z_j))
```

The implementation subtracts the maximum logit before exponentiation:

```
exp(z_i - max(z))
```

This improves numerical stability and prevents unnecessarily large exponentials.

For the EMNIST output layer, softmax produces ten probabilities corresponding to the ten digit classes.

## Layers and Network

A `Layer` combines a dense layer with an activation function.

Conceptually:

```
Layer
  |
  +-- Dense
  |     |
  |     +-- Weights
  |     +-- Biases
  |
  +-- Activation
  |
  +-- Cached values
        |
        +-- Input
        +-- Z
        +-- Gradients
```

A `Network` contains multiple layers.

The architecture is configured using `LayerConfig` structures instead of hardcoding each layer.

For example:

```
784 -> 128 -> 64 -> $(class_size)
```

can be configured as:

```
Layer 0:
    input_size  = 784
    output_size = 128
    activation  = ReLU

Layer 1:
    input_size  = 128
    output_size = 64
    activation  = ReLU

Layer 2:
    input_size  = 64
    output_size = $(class_size)
    activation  = Softmax + Cross-Entropy
```

## Forward Propagation

For each layer, the forward pass is:

```
X
|
v
Z = XW + b
|
v
A = activation(Z)
|
v
next layer
```

For a complete network:

```
Input
  |
  v
Dense
  |
  v
ReLU
  |
  v
Dense
  |
  v
ReLU
  |
  v
Dense
  |
  v
Softmax
  |
  v
Prediction
```

The final output is a matrix where each row represents one sample and each column represents a class probability.

## Loss Functions

The project implements Mean Squared Error and Cross-Entropy.

### Mean Squared Error

MSE is defined as:

```
L = (1 / n) * sum((prediction_i - target_i)^2)
```

Its derivative with respect to the prediction is:

```
dL/dA = 2(A - Y) / n
```

MSE is useful for testing and understanding gradients, but it is not the preferred loss function for multi-class classification.

### Cross-Entropy

For a prediction vector `P` and one-hot target `Y`:

```
L = -sum(Y_i * log(P_i))
```

For a one-hot encoded target, only the probability assigned to the correct class contributes to the loss.

For example:

```
Prediction:
    [0.01, 0.02, 0.90, 0.01, ...]

Target:
    [0, 0, 1, 0, ...]
```

The loss is approximately:

```
-log(0.90)
```

The implementation clamps probabilities before taking the logarithm to avoid `log(0)`.

### Softmax + Cross-Entropy

The network has a combined backward path for softmax followed by cross-entropy.

For:

```
A = softmax(Z)
```

and cross-entropy loss, the derivative simplifies to:

```
dZ = A - Y
```

When training with a batch, the gradient is averaged:

```
dZ = (A - Y) / batch_size
```

This avoids explicitly constructing the full softmax Jacobian and is both simpler and numerically appropriate for classification.

## Backpropagation

Backpropagation calculates how much each parameter contributed to the loss.

The network traverses its layers in reverse order.

The general flow is:

```
Loss
  |
  v
Output layer
  |
  v
Hidden layer
  |
  v
Hidden layer
  |
  v
Input
```

For a dense layer:

```
Z = XW + b
```

Given `dZ`, the gradients are:

```
dW = X^T dZ

db = sum(dZ)

dX = dZ W^T
```

Where:

- `dW` = gradient of the weights
- `db` = gradient of the biases
- `dX` = gradient passed to the previous layer

The `dX` calculated by one layer becomes the `dA` input gradient for the previous layer.

Activation functions then transform that gradient according to their derivatives.

### Activation Backward Pass

For ReLU:

```
dZ = dA              if Z > 0
     0               otherwise
```

For Leaky ReLU:

```
dZ = dA              if Z > 0
     0.01 * dA       otherwise
```

For sigmoid:

```
s = sigmoid(Z)

dZ = dA * s * (1 - s)
```

For tanh:

```
t = tanh(Z)

dZ = dA * (1 - t^2)
```

For the Softmax + Cross-Entropy combination:

```
dZ = dA
```

where `dA` has already been calculated as:

```
prediction - target
```

This is why the final activation can use a specialized backward path for cross-entropy.

## Optimization

The project currently implements Stochastic Gradient Descent (SGD).

The basic update rule is:

```
W = W - learning_rate * dW

b = b - learning_rate * db
```

The optimizer is responsible for applying these updates to every trainable layer in the network.

The learning rate controls the size of each parameter update.

- A learning rate that is too large can cause unstable training.
- A learning rate that is too small can make training unnecessarily slow.

## Mini-Batch Training

The network can train using batches of samples rather than processing the entire dataset at once.

For a batch:

- `X` = batch of input samples
- `Y` = batch of targets

The network performs:

```
X
  |
  v
Forward pass
  |
  v
Predictions
  |
  v
Loss
  |
  v
Backward pass
  |
  v
Gradients
  |
  v
SGD update
```

Using batches reduces memory requirements and allows matrix operations to process multiple samples efficiently.

## EMNIST

The project uses the EMNIST handwritten dataset.

Each image is:

```
28 x 28 pixels
```

The image is flattened into:

```
784 values
```

Each pixel is normalized from:

```
0 - 255
```

into:

```
0.0 - 1.0
```

The output layer produces a probability for each digit.

The predicted class is the index of the largest probability.

For example(for mnist):

```
[0.01, 0.02, 0.03, 0.91, 0.01, ...]
```

The prediction is:

```
3
```

### EMNIST Data Loader

The EMNIST loader reads the original IDX binary format.

EMNIST image files contain:

- Magic number
- Number of images
- Number of rows
- Number of columns
- Pixel data

EMNIST label files contain:

- Magic number
- Number of labels
- Label data

The loader handles the big-endian values stored in the IDX headers.

Images can be loaded directly as normalized floating-point values. The normalized representation is:

```
pixel / 255.0
```

## Inference

After training, the network can be used without performing any parameter updates.

Inference consists only of:

```
Input image
    |
    v
Network forward pass
    |
    v
Output probabilities
    |
    v
Argmax
    |
    v
Predicted digit
```

For a single EMNIST image:

```
Matrix input = 1 x 784
```

The network produces:

```
Matrix output = 1 x $(class_size)
```

The largest value in the output represents the predicted digit. The corresponding probability can be used as the model's confidence estimate.

## EMNIST Evaluation

The test dataset can be passed through the trained network to calculate classification accuracy.

```
accuracy = correct_predictions / total_predictions
```

For example:

```
Correct: 9200
Total:   10000

Accuracy = 92.00%
```

## Rendering

The project includes an SDL2-based EMNIST renderer.

A normalized `1 x 784` image can be displayed as a `28 x 28` image.

The renderer converts normalized floating-point values back into grayscale pixel values:

```
0.0 -> 0
1.0 -> 255
```

The predicted and actual labels can also be included in the window title. This makes it possible to visually inspect individual predictions instead of relying only on numerical accuracy.

## Memory Management

The project is written in C, so memory ownership is explicit.

Matrices allocate their own data and must eventually be destroyed.

Typical lifetime:

```c
Matrix matrix = matrix_empty();

matrix_create(...);

/* use matrix */

matrix_destroy(&matrix);
```

The same principle applies to:

- Network layers
- Optimizers
- EMNIST image buffers
- EMNIST label buffers

The project uses explicit cleanup rather than relying on automatic memory management.

## Error Handling

The project uses assertions for programmer errors and explicit return values for operations that can legitimately fail.

For example, matrix allocation functions can return failure:

```c
if (!matrix_create(...)) {
    ...
}
```

Assertions are used to verify assumptions such as:

- Valid pointers
- Compatible matrix dimensions
- Valid matrix sizes
- Valid labels
- Valid layer configurations

Assertions can be disabled for release builds using `NDEBUG`.

## Building

The project uses CMake.

Configure the build:

```bash
cmake -S . -B build
```

Build the project:

```bash
cmake --build build
```

Run the executable from the project root so that the relative EMNIST paths resolve correctly:

```bash
./nn
```

The exact executable location depends on the CMake configuration.

## EMNIST Dataset

The project also includes a script for downloading the dataset:

```
script/download_emnist.sh
```

Run:

```bash
./script/download_emnist.sh $(emnist_type)
```

## Design Philosophy

This project intentionally avoids machine-learning frameworks.

The goal is not to reproduce the performance or feature set of libraries such as PyTorch or TensorFlow.

The goal is to understand what those libraries are doing underneath.

The important operations are implemented directly:

```
Matrix multiplication
    |
    v
Dense transformation
    |
    v
Activation
    |
    v
Loss
    |
    v
Gradient calculation
    |
    v
Backpropagation
    |
    v
Parameter update
    |
    v
Learning
```

This makes the mathematical relationship between the forward pass, loss, gradients, and parameter updates explicit.

## Current EMNIST Network

A typical EMNIST configuration is:

```
Input
    784
    |
    v
Dense
    784 -> 128
    |
    v
ReLU
    |
    v
Dense
    128 -> $(class_size)
    |
    v
Softmax + Cross-Entropy
    |
    v
Output
```

The network learns to map the 784 pixel values of an EMNIST image to a probability distribution over the ten digit classes.

## Project Goal

The ultimate goal is to build a complete neural-network training system in C while understanding every major component involved.

The project is intentionally developed from the bottom up:

```
Low-level matrix operations
        |
        v
Neural network layers
        |
        v
Forward propagation
        |
        v
Loss functions
        |
        v
Backpropagation
        |
        v
Optimization
        |
        v
EMNIST training
        |
        v
Evaluation and inference
```

The emphasis is on understanding the implementation and mathematics rather than simply obtaining a working classifier.