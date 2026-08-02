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
#include "loss.h"
#include "matrix.h"

#include <math.h>
#include <assert.h>

/*==============================================================================
 * Loss Functions
 *============================================================================*/

float loss_mse(const Matrix *prediction, const Matrix *target) {
    assert(prediction);
    assert(target);
    assert(prediction->data);
    assert(target->data);
    assert(prediction->rows == target->rows);
    assert(prediction->cols == target->cols);

    usize n = matrix_size(prediction);
    assert(n > 0);

    float sum = 0.0f;
    for (usize i = 0; i < n; i++) {
        float diff = prediction->data[i] - target->data[i];
        sum += diff * diff;
    }

    return sum / (float)n;
}

float loss_cross_entropy(const Matrix *prediction, const Matrix *target) {
    assert(prediction);
    assert(target);
    assert(prediction->data);
    assert(target->data);
    assert(prediction->rows == target->rows);
    assert(prediction->cols == target->cols);

    usize n = matrix_size(prediction);
    assert(n > 0);

    float sum = 0.0f;
    for (usize i = 0; i < n; i++) {
        /* Clamp prediction to avoid log(0) numerical instability */
        float p = fmaxf(prediction->data[i], 1e-7f);
        sum += target->data[i] * logf(p);
    }

    return -sum;
}