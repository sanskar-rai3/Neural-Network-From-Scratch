# Neural Network From Scratch in C

A neural network implementation written from scratch in **C11**, with no machine-learning frameworks.

The goal of this project is to understand how neural networks actually work by implementing the core components manually — matrix operations, forward propagation, backpropagation, loss functions, optimizers, and eventually MNIST training.

## Current Status

### Implemented

- [x] Matrix library
  - [x] Matrix creation/destruction
  - [x] Matrix multiplication
  - [x] Matrix addition/subtraction
  - [x] Hadamard product
  - [x] Scalar multiplication
  - [x] Transposition
  - [x] Row broadcasting
  - [x] Random initialization
  - [x] He initialization
  - [x] Xavier initialization

- [x] Activation functions
  - [x] ReLU
  - [x] Leaky ReLU
  - [x] Sigmoid
  - [x] Tanh
  - [x] Softmax
  - [x] Activation backpropagation

- [x] Dense layers
  - [x] Forward propagation
  - [x] Backpropagation
  - [x] Weight gradients
  - [x] Bias gradients
  - [x] Input caching

- [x] Network
  - [x] Sequential layers
  - [x] Forward propagation
  - [x] Backpropagation

- [x] Loss functions
  - [x] Mean Squared Error
  - [x] Categorical Cross-Entropy

- [x] Optimizer
  - [x] SGD

- [x] MNIST
  - [x] MNIST data loader
  - [x] MNIST sample renderer

- [x] XOR training
  - [x] Successfully trains a neural network to learn XOR

## XOR Example

The current implementation can learn the XOR function:

```text
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0n from scratch in C
