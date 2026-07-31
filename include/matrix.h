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

#ifndef __MATRIX_H_
#define __MATRIX_H_

#include "common.h"

typedef struct Matrix {
    usize rows;
    usize cols;
    float *data;
} Matrix;

/* Creation & Destruction */
int matrix_create(Matrix *mat, usize rows, usize cols);
void matrix_free(Matrix *mat);

/* Element Access */
float matrix_get(const Matrix *mat, usize row, usize col);
void matrix_set(Matrix *mat, usize row, usize col, float value);

/* Utility */
void matrix_fill(Matrix *mat, float value);
void matrix_randomize(Matrix *mat, float min, float max);
void matrix_copy(Matrix *dest, const Matrix *src);

/* Arithmetic */
Matrix matrix_add(const Matrix *a, const Matrix *b);
Matrix matrix_subtract(const Matrix *a, const Matrix *b);
Matrix matrix_multiply(const Matrix *a, const Matrix *b);
Matrix matrix_hadamard(const Matrix *a, const Matrix *b);
Matrix matrix_scalar_multiply(const Matrix *mat, float scalar);
Matrix matrix_transpose(const Matrix *mat);

/* In-place Arithmetic */
void matrix_add_inplace(Matrix *a, const Matrix *b);
void matrix_subtract_inplace(Matrix *a, const Matrix *b);
void matrix_scalar_multiply_inplace(Matrix *mat, float scalar);

/* Element-wise Operations */
void matrix_apply(Matrix *mat, float (*func)(float));

#endif
