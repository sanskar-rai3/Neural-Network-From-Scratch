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

#ifndef __LAYER_H__
#define __LAYER_H__

#include "common.h"
#include "dense.h"
#include "activation.h"

/*==============================================================================
 * Types & Configuration
 *============================================================================*/

/**
 * @brief High-level neural network layer combining a Dense layer with an activation function.
 */
typedef struct Layer {
    Dense dense;               /**< Underlying fully connected dense layer (weights & biases) */
    ActivationType activation; /**< Activation function applied post-linear transformation */
} Layer;

/**
 * @brief Configuration parameters used to initialize a Layer.
 */
typedef struct LayerConfig {
    usize input_size;          /**< Number of input features */
    usize output_size;         /**< Number of output neurons/features */
    ActivationType activation; /**< Type of activation function to apply */
} LayerConfig;


/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

/**
 * @brief Initializes a Layer according to the specified configuration.
 *
 * @param layer  Pointer to the Layer structure to initialize.
 * @param config Pointer to the layer configuration parameters.
 * @return       1 on success, 0 on failure.
 */
int layer_init(Layer *layer, const LayerConfig *config);

/**
 * @brief Frees all allocated memory owned by the layer.
 *
 * @param layer Pointer to the Layer to destroy.
 */
void layer_destroy(Layer *layer);


/*==============================================================================
 * Forward Pass
 *============================================================================*/

/**
 * @brief Performs a complete forward pass through the layer: output = activation(input * W + b).
 *
 * @param output Destination matrix to store the results (batch_size x output_size).
 * @param layer  Pointer to the initialized Layer.
 * @param input  Input matrix (batch_size x input_size).
 */
void layer_forward(Matrix *output, const Layer *layer, const Matrix *input);

#endif