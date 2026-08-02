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

#ifndef __DENSE_H__
#define __DENSE_H__

#include "common.h"
#include "matrix.h"

/**
 * @brief Fully connected (Dense) layer structure containing weights and biases.
 */
typedef struct Dense {
    Matrix weights; /**< Weight matrix of shape (input_size x output_size) */
    Matrix bias;    /**< Bias matrix of shape (1 x output_size) */
} Dense;

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

/**
 * @brief Initializes a dense layer with allocated weights and biases.
 * 
 * Weights are typically initialized using He or Xavier uniform initialization,
 * and biases are zero-initialized.
 * 
 * @param layer       Pointer to the Dense layer structure to initialize.
 * @param input_size  Number of input features.
 * @param output_size Number of output features/neurons.
 * @return            1 on success, 0 on failure.
 */
int dense_init(Dense *layer, usize input_size, usize output_size);

/**
 * @brief Frees all memory associated with a dense layer's weights and biases.
 * 
 * @param layer Pointer to the Dense layer to destroy.
 */
void dense_destroy(Dense *layer);

/*==============================================================================
 *  Forward Pass
 *============================================================================*/

/**
 * @brief Performs a forward pass through the dense layer: output = (input * weights) + bias.
 * 
 * @param output Destination matrix to store the result. Must be pre-allocated 
 *               with shape (batch_size x output_size).
 * @param layer  Pointer to the initialized Dense layer.
 * @param input  Input matrix of shape (batch_size x input_size).
 */
void dense_forward(Matrix *output, const Dense *layer, const Matrix *input);

#endif