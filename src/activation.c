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
#include "math/matrix.h"

#include <math.h>
#include <assert.h>

/*==============================================================================
 * static constant
 *============================================================================*/

static const float LEAKY_RELU_ALPHA = 0.01f;

/*==============================================================================
 * Matrix Activation Functions
 *============================================================================*/

static void softmax(Matrix *mat) {
    assert(mat && mat->data);
    assert(mat->rows > 0 && mat->cols > 0);

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
        float inv_sum = 1.0f / (sum + 1e-7f); /* Avoid div by zero */
        for (usize c = 0; c < mat->cols; c++) {
            mat->data[offset + c] *= inv_sum;
        }
    }
}

/*==============================================================================
 * Forward Pass 
 *============================================================================*/

void activation_forward(Matrix *mat, ActivationType type) {
    assert(mat != NULL && mat->data != NULL);

    usize total = matrix_size(mat);

    switch (type) {
        case ACT_RELU: {
            for (usize i = 0; i < total; i++) {
                if (mat->data[i] < 0.0f) mat->data[i] = 0.0f;
            }
            break;
        }

        case ACT_LEAKY_RELU: {
            for (usize i = 0; i < total; i++) {
                if (mat->data[i] < 0.0f) mat->data[i] *= LEAKY_RELU_ALPHA;
            }
            break;
        }

        case ACT_TANH: {
            for (usize i = 0; i < total; i++) {
                mat->data[i] = tanhf(mat->data[i]);
            }
            break;
        }

        case ACT_SIGMOID: {
            for (usize i = 0; i < total; i++) {
                mat->data[i] = 1.0f / (1.0f + expf(-mat->data[i]));
            }
            break;
        }

        case ACT_SOFTMAX_CEL:
            softmax(mat);
            break;

        case ACT_NONE:
        default:
            break;
    }
}

void activation_backward(Matrix *dZ, const Matrix *dA, const Matrix *Z, ActivationType type) {
    assert(dZ != NULL);
    assert(dA != NULL);
    assert(Z != NULL);
    assert(dA->rows == Z->rows && dA->cols == Z->cols);
    assert(dZ->rows == Z->rows && dZ->cols == Z->cols);

    usize total = Z->rows * Z->cols;

    switch (type) {
        case ACT_RELU: {
            /* ReLU derivative: 1 if Z > 0, else 0 */
            for (usize i = 0; i < total; i++) {
                dZ->data[i] = (Z->data[i] > 0.0f) ? dA->data[i] : 0.0f;
            }
            break;
        }

        case ACT_LEAKY_RELU: {
            for (usize i = 0; i < total; i++) {
                float slope = (Z->data[i] > 0.0f) ? 1.0f : LEAKY_RELU_ALPHA;
                dZ->data[i] = dA->data[i] * slope;
            }
            break;
        }

        case ACT_TANH: {
            /* Tanh derivative: 1 - tanh^2(Z) */
            for (usize i = 0; i < total; i++) {
                float t = tanhf(Z->data[i]);
                dZ->data[i] = dA->data[i] * (1.0f - t * t);
            }
            break;
        }

        case ACT_SIGMOID: {
            /* Sigmoid derivative: s(Z) * (1 - s(Z)) */
            for (usize i = 0; i < total; i++) {
                float s = 1.0f / (1.0f + expf(-Z->data[i]));
                dZ->data[i] = dA->data[i] * (s * (1.0f - s));
            }
            break;
        }

        case ACT_SOFTMAX_CEL: {
            /*
             * Softmax + Cross-Entropy uses a fused backward pass.
             *
             * The loss function computes:
             *
             *     dZ = (A - Y) / batch_size
             *
             * Therefore, dA is already the gradient with respect
             * to the logits and is passed through unchanged.
             */
            for (usize i = 0; i < total; i++) {
                dZ->data[i] = dA->data[i];
            }

            break;
        }
                              
        case ACT_NONE:
        default: {
            /* Linear activation pass-through */
            for (usize i = 0; i < total; i++) {
                dZ->data[i] = dA->data[i];
            }
            break;
        }
    }
}
