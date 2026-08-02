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
#include "activation.h"
#include "matrix.h"

#include <math.h>
#include <assert.h>

/*==============================================================================
 * Scalar Activation Callbacks (for matrix_apply)
 *============================================================================*/

static float relu_scalar(float x) {
    return x < 0.0f ? 0.0f : x;
}

static float leaky_relu_scalar(float x) {
    return x < 0.0f ? 0.01f * x : x;
}

static float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/*==============================================================================
 * Matrix Activation Functions
 *============================================================================*/

static void softmax(Matrix *mat) {
    assert(mat);
    assert(mat->data);
    assert(mat->rows > 0);
    assert(mat->cols > 0);

    /* Process softmax row by row to support multi-sample batch processing */
    for (usize r = 0; r < mat->rows; r++) {
        usize offset = r * mat->cols;

        /* Find max value in row for numerical stability */
        float max_val = mat->data[offset];
        for (usize c = 1; c < mat->cols; c++) {
            if (mat->data[offset + c] > max_val) {
                max_val = mat->data[offset + c];
            }
        }

        /* Exponentiate and sum */
        float sum = 0.0f;
        for (usize c = 0; c < mat->cols; c++) {
            mat->data[offset + c] = expf(mat->data[offset + c] - max_val);
            sum += mat->data[offset + c];
        }

        /* Normalize */
        for (usize c = 0; c < mat->cols; c++) {
            mat->data[offset + c] /= sum;
        }
    }
}

/*==============================================================================
 * Activation Application
 *============================================================================*/

void activation_apply(Matrix *mat, ActivationType activation) {
    assert(mat);
    assert(mat->data);

    switch (activation) {
        case ACT_RELU:
            matrix_apply(mat, relu_scalar);
            break;
        case ACT_LEAKY_RELU:
            matrix_apply(mat, leaky_relu_scalar);
            break;
        case ACT_TANH:
            matrix_apply(mat, tanhf);
            break;
        case ACT_SIGMOID:
            matrix_apply(mat, sigmoid_scalar);
            break;
        case ACT_SOFTMAX:
            softmax(mat);
            break;
        case ACT_NONE:
            break;
        default:
            assert(!"Invalid activation function");
            break;
    }
}