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

int dense_init(Dense *layer, usize input_size, usize output_size) {
    if (!matrix_create(&layer->weights, input_size, output_size)) {
        return 0;
    }

    if (!matrix_create(&layer->bias, 1, output_size)) {
        return 0;
    }

    matrix_randomize(&layer->weights, -1.0f, 1.0f);

    matrix_fill(&layer->bias, 0.0f);

    return 1;
}

void dense_destroy(Dense *layer) {
    matrix_destroy(&layer->weights); 
    matrix_destroy(&layer->bias); 
}

Matrix dense_forward(Dense *layer, const Matrix *input) {
    Matrix output = matrix_multiply(input, &layer->weights);

    for (usize r = 0; r < output.rows; r++)
        for (usize c = 0; c < output.cols; c++)
            matrix_set(
                &output,
                r,
                c,
                matrix_get(&output, r, c) +
                matrix_get(&layer->bias, 0, c)
            );

    return output;
}
