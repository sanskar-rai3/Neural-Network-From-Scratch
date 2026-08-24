# Neural Network From Scratch in C

[![License: GPL v3 or later](https://img.shields.io/badge/License-GPL%20v3%20or%20later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

A feed-forward neural network implemented entirely in C, without machine-learning frameworks. The goal is to understand how neural networks work internally by implementing the underlying mathematics and components directly rather than hiding them behind high-level libraries.

Supports matrix operations, MNIST loading, dense layers, activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax), loss functions (MSE, Cross-Entropy), backpropagation, configurable multi-layer networks, SGD optimization, MNIST training/inference/evaluation, and SDL2-based sample rendering.

## Documentation

Full API and design documentation: https://sanskar-rai3.github.io/Neural-Network-From-Scratch/

## Building

```bash
cmake -S . -B build
cmake --build build
```

Run from the project root so relative MNIST paths resolve:

```bash
./nn
```

## MNIST Dataset

Expected files:

```
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

Download with:

```bash
./script/download_mnist.sh
```

## License
 
This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
 
You may use, modify, and redistribute this software under the terms of the GPL-3.0-or-later license.
 
See the LICENSE file for the full license text.
