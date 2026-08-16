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

    matrix_create(&layer->weights, input_size, output_size);
    matrix_create(&layer->bias, 1, output_size);

    matrix_he_uniform(&layer->weights, input_size);
    matrix_fill(&layer->bias, 0.0f);

    return 1;
}

void dense_destroy(Dense *layer) {
    assert(layer);

    matrix_destroy(&layer->weights);
    matrix_destroy(&layer->bias);
}

/*==============================================================================
 * Forward Pass
 *============================================================================*/

void dense_forward(Matrix *output, const Dense *layer, const Matrix *input) {
    assert(output);
    assert(layer);
    assert(input);
    assert(input->cols == layer->weights.rows);
    assert(output->rows == input->rows);
    assert(output->cols == layer->weights.cols);

    matrix_multiply(output, input, &layer->weights);
    matrix_add_row_inplace(output, &layer->bias);
}
