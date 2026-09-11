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

#include "common.h"
#include "layer.h"
#include "layer/dense.h"
#include "matrix.h"

#include <assert.h>

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

int layer_init(Layer *layer, const LayerConfig *config) {
    assert(layer);
    assert(config);

    switch (config->layer_type) {
        case LAYER_DENSE: {
            layer->layer_type = LAYER_DENSE;

            if (!dense_init(&layer->dense, &config->dense_config)) {
                return 0;
            }

            layer->activation = config->activation;
        }

        default:
            break;
    }

    layer->z_cache = matrix_empty();

    return 1;
}

void layer_destroy(Layer *layer) {
    assert(layer);

    dense_destroy(&layer->dense);
    matrix_destroy(&layer->z_cache);

    layer->activation = ACT_NONE;
}

/*==============================================================================
 * Forward Pass
 *============================================================================*/

void layer_forward(Matrix *output, Layer *layer, const Matrix *input) {
    assert(output);
    assert(layer);
    assert(input);

    assert(input->cols == layer->dense.weights.rows);
    assert(output->rows == input->rows);
    assert(output->cols == layer->dense.weights.cols);

    switch (layer->layer_type) {
        case LAYER_DENSE: {
            /* Z = XW + b */
            dense_forward(output, &layer->dense, input);

            /* Cache Z for the backward pass */
            if (layer->z_cache.rows != output->rows ||
                layer->z_cache.cols != output->cols) {

                matrix_destroy(&layer->z_cache);

                matrix_create(&layer->z_cache, output->rows, output->cols);
            }

            matrix_copy(&layer->z_cache, output);

            /* A = activation(Z) */
            activation_forward(output, layer->activation);

            break;
        }

        default:
            break;
    }
}

/*==============================================================================
 *  Backward Pass
 *============================================================================*/

void layer_backward(Matrix *d_input, Layer *layer, const Matrix *d_output) {
    assert(d_input);
    assert(layer);
    assert(d_output);

    assert(d_output->rows == layer->z_cache.rows);
    assert(d_output->cols == layer->z_cache.cols);

    Matrix dZ;

    matrix_create(
        &dZ,
        layer->z_cache.rows,
        layer->z_cache.cols
    );

    /*
     * Activation backward:
     *
     * dZ = dA ⊙ f'(Z)
     *
     * d_output = dA
     */
    activation_backward(
        &dZ,
        d_output,
        &layer->z_cache,
        layer->activation
    );

    /*
     * Dense backward:
     *
     * dX = dZ * W^T
     * dW = X^T * dZ
     * db = sum(dZ)
     */
    dense_backward(
        d_input,
        &layer->dense,
        &dZ
    );

    matrix_destroy(&dZ);
}
