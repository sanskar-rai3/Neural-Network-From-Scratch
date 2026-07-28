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

#include <math.h>
#include <stdlib.h>

/* Helper Function */
static MATRIX matrix_empty(void) {
    return (MATRIX) {
        .rows = 0,
        .cols = 0,
        .data = NULL
    };
}

/* Creation & Destruction */
int matrix_create(MATRIX *mat, usize rows, usize cols) {
    mat->data = malloc(rows * cols * sizeof(float));
    if (!mat->data) {
        return 0;
    }

    mat->rows = rows;
    mat->cols = cols;

    return 1;
}

void matrix_free(MATRIX *mat) {
    free(mat->data);

    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
}

/* Element Access */
float matrix_get(const MATRIX *mat, usize row, usize col) {
    return mat->data[row * mat->cols + col];
}

void matrix_set(MATRIX *mat, usize row, usize col, float value) {
    mat->data[row * mat->cols + col] = value;
}

/* Utility */
void matrix_fill(MATRIX *mat, float value);
void matrix_randomize(MATRIX *mat, float min, float max);
void matrix_copy(MATRIX *dest, const MATRIX *src);
void matrix_print(const MATRIX *mat);

/* Arithmetic */
MATRIX matrix_add(const MATRIX *a, const MATRIX *b) {
    MATRIX result;

    if (a->rows != b->rows || a->cols != b->cols) {
        return matrix_empty();
    }

    if (!matrix_create(&result, a->rows, a->cols)) {
        return matrix_empty();
    }

    for (usize i = 0; i < result.rows * result.cols; i++) {
        result.data[i] = a->data[i] + b->data[i];
    }

    return result;
}

MATRIX matrix_subtract(const MATRIX *a, const MATRIX *b) {
    MATRIX result;

    if (a->rows != b->rows || a->cols != b->cols) {
        return matrix_empty();
    }

    if (!matrix_create(&result, a->rows, a->cols)) {
        return matrix_empty();
    }

    for (usize i = 0; i < result.rows * result.cols; i++) {
        result.data[i] = a->data[i] - b->data[i];
    }

    return result;
}

MATRIX matrix_multiply(const MATRIX *a, const MATRIX *b) {

}

MATRIX matrix_hadamard(const MATRIX *a, const MATRIX *b) {

}

MATRIX matrix_scalar_multiply(const MATRIX *mat, float scalar) {
    MATRIX result;
    if (!matrix_create(&result, mat->rows, mat->cols)) {
        return matrix_empty();
    }

    for (usize i = 0; i < mat->rows * mat->cols; i++) {
        result.data[i] = mat->data[i] * scalar;
    }

    return result;
}

MATRIX matrix_transpose(const MATRIX *mat) {

}

/* In-place Arithmetic */
void matrix_add_inplace(MATRIX *a, const MATRIX *b);
void matrix_subtract_inplace(MATRIX *a, const MATRIX *b);
void matrix_scalar_multiply_inplace(MATRIX *mat, float scalar);

/* Element-wise Operations */
void matrix_apply(MATRIX *mat, float (*func)(float));