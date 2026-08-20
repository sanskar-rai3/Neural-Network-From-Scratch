/*
 * Neural Network From Scratch in C
 *
 * Copyright (C) 2026 Sanskar Rai 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "common.h"
#include "matrix.h"

/*==============================================================================
 * Types & Enumerations
 *============================================================================*/

/**
 * @brief Supported neural network activation function types.
 */
typedef enum ActivationType {
    ACT_NONE = 0,           /**< Identity function / Linear pass-through: f(x) = x */
    ACT_RELU,               /**< Rectified Linear Unit: f(x) = max(0, x) */
    ACT_LEAKY_RELU,         /**< Leaky ReLU: f(x) = x if x > 0 else 0.01 * x */
    ACT_TANH,               /**< Hyperbolic Tangent: f(x) = tanh(x) */
    ACT_SIGMOID,            /**< Sigmoid / Logistic: f(x) = 1 / (1 + exp(-x)) */
    ACT_SOFTMAX_CEL         /**< Softmax (row-wise / batch probability distribution) with Cross-Entropy Loss */
} ActivationType;


/*==============================================================================
 * Forward pass 
 *============================================================================*/

/**
 * @brief Applies the specified activation function to a matrix in-place.
 *
 * For scalar activations (ReLU, Leaky ReLU, Tanh, Sigmoid), the function is 
 * applied element-wise. For Softmax, it is evaluated across rows (samples).
 *
 * @param mat        Pointer to the matrix to modify in-place.
 * @param activation The activation function type to apply.
 */
void activation_forward(Matrix *mat, ActivationType activation);

/*==============================================================================
 * Backward pass 
 *============================================================================*/

/**
 * @brief Computes pre-activation gradient: dZ = dA * derivative(Z).
 *
 * @param dZ   Output matrix to store pre-activation gradient (same shape as Z).
 * @param dA   Incoming output gradient from next layer (same shape as Z).
 * @param Z    Stored pre-activation cache from forward pass (batch_size x cols).
 * @param type Activation function type.
 */
void activation_backward(Matrix *dZ, const Matrix *dA, const Matrix *Z, ActivationType type);

#endif
