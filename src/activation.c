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

/* Helper Function */
static void ReLU(Matrix *mat) {
    assert(mat != NULL);
    for (usize i = 0; i < matrix_size(mat); i++) {
        mat->data[i] = mat->data[i] < 0 ? 0 : mat->data[i];
    }
}

static void Leaky_ReLU(Matrix *mat) {
    assert(mat != NULL);
    for (usize i = 0; i < matrix_size(mat); i++) {
        mat->data[i] = mat->data[i] < 0 ? 0.01 * mat->data[i] : mat->data[i];
    }
}

static void Tanh(Matrix *mat) {
    assert(mat != NULL);
    for (usize i = 0; i < matrix_size(mat); i++) {
        mat->data[i] = tanhf(mat->data[i]);
    }
}

static void Sigmoid(Matrix *mat) {
    assert(mat != NULL);
    for (usize i = 0; i < matrix_size(mat); i++) {
        mat->data[i] = 1.0f / (1.0f + expf(-mat->data[i]));
    }
}

static void Softmax(Matrix *mat) {
    assert(mat != NULL);
    usize size = matrix_size(mat);

    float max = mat->data[0];
    for (usize i = 1; i < size; i++) {
        if (mat->data[i] > max) {
            max = mat->data[i];
        }
    }     

    float sum = 0.0f;
    for (usize i = 0; i < size; i++) {
        mat->data[i] = expf(mat->data[i] - max);
        sum += mat->data[i];
    }

    for (usize i = 0; i < size; i++) {
        mat->data[i] /= sum;
    }
}

/* Apply Function */
void activation_apply(Matrix *mat, Activation activation) {
    switch (activation) {
        case RELU:         ReLU(mat); break;
        case LEAKY_RELU:   Leaky_ReLU(mat); break;
        case TANH:         Tanh(mat); break;
        case SIGMOID:      Sigmoid(mat); break;
        case SOFTMAX:      Softmax(mat); break;
        default:           assert(!"Invalid activation function"); break;
    }
}