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
#include "dense.h"
#include "matrix.h"

#include <assert.h>

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

int layer_init(Layer *layer, const LayerConfig *config) {
    assert(layer);
    assert(config);
    assert(config->input_size > 0);
    assert(config->output_size > 0);

    if (!dense_init(&layer->dense, config->input_size, config->output_size)) {
        return 0;
    }

    layer->activation = config->activation;

    return 1;
}

void layer_destroy(Layer *layer) {
    assert(layer);

    dense_destroy(&layer->dense);
    layer->activation = ACT_NONE;
}

/*==============================================================================
 * Forward Pass
 *============================================================================*/

void layer_forward(Matrix *output, const Layer *layer, const Matrix *input) {
    assert(output);
    assert(layer);
    assert(input);
    assert(input->cols == layer->dense.weights.rows);
    assert(output->rows == input->rows);
    assert(output->cols == layer->dense.weights.cols);

    /* Perform linear transformation: output = (input * weights) + bias */
    dense_forward(output, &layer->dense, input);

    /* Apply activation function in-place on output matrix */
    activation_apply(output, layer->activation);
}