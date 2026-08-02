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
#include "matrix.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

/* Helper Function */
static inline Matrix matrix_empty(void) {
    return (Matrix) {
        .rows = 0,
        .cols = 0,
        .data = NULL
    };
}

static inline float random_float(float min, float max) {
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

/* Creation & Destruction */
int matrix_create(Matrix *mat, usize rows, usize cols) {
    mat->data = malloc(rows * cols * sizeof(float));
    if (!mat->data) {
        return 0;
    }

    mat->rows = rows;
    mat->cols = cols;

    return 1;
}

int matrix_copy_from_buffer(Matrix *mat, usize rows, usize cols, float *buffer) {
    mat->data = malloc(rows * cols * sizeof(float));
    if (!mat->data) {
        return 0;
    }

    memcpy(mat->data, buffer, rows * cols * sizeof(float));

    mat->rows = rows;
    mat->cols = cols;

    return 1;
}

void matrix_destroy(Matrix *mat) {
    free(mat->data);

    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
}

/* Element Access */
float matrix_get(const Matrix *mat, usize row, usize col) {
    return mat->data[row * mat->cols + col];
}

void matrix_set(Matrix *mat, usize row, usize col, float value) {
    mat->data[row * mat->cols + col] = value;
}

/* Utility */
usize matrix_size(const Matrix *mat) {
    return mat->rows * mat->cols;
}

Matrix matrix_init(void) {
    return (Matrix) {
        .rows = 0,
        .cols = 0,
        .data = NULL
    };
}

void matrix_fill(Matrix *mat, float value) {
    for (usize i = 0; i < matrix_size(mat) ; i++) {
        mat->data[i] = value;
    }
}

void matrix_copy(Matrix *dest, const Matrix *src) {
    if (dest->rows != src->rows || dest->cols != src->cols) {
        return;
    }
     
    memcpy(dest->data, src->data, matrix_size(dest) * sizeof(float));
}

/* Random */
void matrix_randomize(Matrix *mat, float min, float max) {
    static bool seeded = false;

    if (!seeded) {
        srand(time(NULL));
        seeded = true;
    }

    for (usize i = 0; i < matrix_size(mat); i++) {
        mat->data[i] = random_float(min, max);
    }
}

void matrix_he_uniform(Matrix *mat, usize fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    matrix_randomize(mat, -limit, limit);
}

void matrix_xavier_uniform(Matrix *mat, usize fan_in, usize fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    matrix_randomize(mat, -limit, limit);
}

/* Arithmetic */
Matrix matrix_add(const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return matrix_empty();
    }

    Matrix result;
    if (!matrix_create(&result, a->rows, a->cols)) {
        return matrix_empty();
    }

    for (usize i = 0; i < matrix_size(&result); i++) {
        result.data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Matrix matrix_subtract(const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return matrix_empty();
    }

    Matrix result;
    if (!matrix_create(&result, a->rows, a->cols)) {
        return matrix_empty();
    }

    for (usize i = 0; i < matrix_size(&result); i++) {
        result.data[i] = a->data[i] - b->data[i];
    }

    return result;
}

Matrix matrix_multiply(const Matrix *a, const Matrix *b) {
    if (a->cols != b->rows) {
        return matrix_empty();
    }

    Matrix result;
    if (!matrix_create(&result, a->rows, b->cols)) {
        return matrix_empty();
    }

    for (usize i = 0; i < result.rows; i++) {
        for (usize j = 0; j < result.cols; j++) {
            float sum = 0.0f;

            for (usize k = 0; k < a->cols; k++) {
                sum += matrix_get(a, i, k) * matrix_get(b, k, j);
            }

            matrix_set(&result, i, j, sum);
        }
    }

    return result;
}

Matrix matrix_hadamard(const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return matrix_empty();
    }

    Matrix result;
    if (!matrix_create(&result, a->rows, a->cols)) {
        return matrix_empty();
    }

    for (usize i = 0; i < matrix_size(&result); i++) {
        result.data[i] = a->data[i] * b->data[i];
    }

    return result;   
}

Matrix matrix_scalar_multiply(const Matrix *mat, float scalar) {
    Matrix result;
    if (!matrix_create(&result, mat->rows, mat->cols)) {
        return matrix_empty();
    }

    for (usize i = 0; i < matrix_size(mat); i++) {
        result.data[i] = mat->data[i] * scalar;
    }

    return result;
}

Matrix matrix_transpose(const Matrix *mat) {
    Matrix result;
    if (!matrix_create(&result, mat->cols, mat->rows)) {
        return matrix_empty();
    }

    for (usize r = 0; r < mat->rows; r++) {
        for (usize c = 0; c < mat->cols; c++) {
            matrix_set(&result, c, r, matrix_get(mat, r, c));
        }
    }

    return result;
}

/* In-place Arithmetic */
void matrix_add_inplace(Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return;
    }

    for (usize i = 0; i < a->rows * b->cols; i++) {
        a->data[i] += b->data[i];
    }
}
void matrix_subtract_inplace(Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return;
    }

    for (usize i = 0; i < a->rows * b->cols; i++) {
        a->data[i] -= b->data[i];
    }
}

void matrix_scalar_multiply_inplace(Matrix *mat, float scalar) {
    for (usize i = 0; i < matrix_size(mat); i++) {
        mat->data[i] *= scalar;
    }
}

/* Element-wise Operations */
void matrix_apply(Matrix *mat, float (*func)(float)) {
    for (usize i = 0; i < matrix_size(mat); i++) {
        mat->data[i] = func(mat->data[i]);
    }
}
