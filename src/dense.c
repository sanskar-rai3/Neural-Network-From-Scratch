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
#include "dense.h"
#include "matrix.h"

#include <assert.h>

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

int dense_init(Dense *layer, usize input_size, usize output_size) {
    assert(layer);
    assert(input_size > 0);
    assert(output_size > 0);

    if (!matrix_create(&layer->weights, input_size, output_size))
        return 0;

    if (!matrix_create(&layer->bias, 1, output_size)) {
        matrix_destroy(&layer->weights);
        return 0;
    }

    matrix_he_uniform(&layer->weights, input_size);
    matrix_fill(&layer->bias, 0.0f);

    layer->input_cache = matrix_empty();

    return 1;
}

void dense_destroy(Dense *layer) {
    assert(layer);

    matrix_destroy(&layer->weights);
    matrix_destroy(&layer->bias);
    if (layer->input_cache.data) {
        matrix_destroy(&layer->input_cache);
    }
}

/*==============================================================================
 * Forward Pass
 *============================================================================*/

void dense_forward(Matrix *output, Dense *layer, const Matrix *input) {
    assert(output);
    assert(layer);
    assert(input);

    assert(input->cols == layer->weights.rows);
    assert(output->rows == input->rows);
    assert(output->cols == layer->weights.cols);

    /* Cache X for backward propagation */
    if (layer->input_cache.rows != input->rows ||
        layer->input_cache.cols != input->cols) {
        
        if (layer->input_cache.data) {
            matrix_destroy(&layer->input_cache);
        }

        assert(matrix_create(
            &layer->input_cache,
            input->rows,
            input->cols
        ));
    }

    matrix_copy(&layer->input_cache, input);

    /* Z = XW + b */
    matrix_multiply(output, input, &layer->weights);
    matrix_add_row_inplace(output, &layer->bias);
}

/*==============================================================================
 * Backward Pass
 *============================================================================*/

void dense_backward(
    Matrix *d_input,
    Matrix *d_weights,
    Matrix *d_bias,
    const Dense *layer,
    const Matrix *dZ
) {
    assert(d_input);
    assert(d_weights);
    assert(d_bias);
    assert(layer);
    assert(dZ);

    const Matrix *X = &layer->input_cache;
    const Matrix *W = &layer->weights;

    /* dW = X^T * dZ
     *
     * X:     (batch x input_size)
     * X^T:   (input_size x batch)
     * dZ:    (batch x output_size)
     *
     * dW:    (input_size x output_size)
     */

    Matrix X_T = matrix_empty();

    assert(matrix_create(&X_T, X->cols, X->rows));

    matrix_transpose(&X_T, X);
    matrix_multiply(d_weights, &X_T, dZ);

    matrix_destroy(&X_T);


    /*
     * db = sum(dZ over the batch)
     *
     * dZ:   (batch x output_size)
     * db:   (1 x output_size)
     */

    matrix_fill(d_bias, 0.0f);

    for (usize r = 0; r < dZ->rows; r++) {
        for (usize c = 0; c < dZ->cols; c++) {
            d_bias->data[c] += dZ->data[r * dZ->cols + c];
        }
    }


    /*
     * dX = dZ * W^T
     *
     * dZ:   (batch x output_size)
     * W^T:  (output_size x input_size)
     *
     * dX:   (batch x input_size)
     */

    Matrix W_T = matrix_empty();

    assert(matrix_create(&W_T, W->cols, W->rows));

    matrix_transpose(&W_T, W);
    matrix_multiply(d_input, dZ, &W_T);

    matrix_destroy(&W_T);
}
