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

#ifndef MATRIX_H
#define MATRIX_H

#include "common.h"

typedef struct MATRIX {
    usize rows;
    usize cols;
    float *data;
} MATRIX;

/* Creation & Destruction */
int matrix_create(MATRIX *mat, usize rows, usize cols);
void matrix_free(MATRIX *mat);

/* Element Access */
float matrix_get(const MATRIX *mat, usize row, usize col);
void matrix_set(MATRIX *mat, usize row, usize col, float value);

/* Utility */
void matrix_fill(MATRIX *mat, float value);
void matrix_randomize(MATRIX *mat, float min, float max);
void matrix_copy(MATRIX *dest, const MATRIX *src);
void matrix_print(const MATRIX *mat);

/* Arithmetic */
MATRIX matrix_add(const MATRIX *a, const MATRIX *b);
MATRIX matrix_subtract(const MATRIX *a, const MATRIX *b);
MATRIX matrix_multiply(const MATRIX *a, const MATRIX *b);
MATRIX matrix_hadamard(const MATRIX *a, const MATRIX *b);
MATRIX matrix_scalar_multiply(const MATRIX *mat, float scalar);
MATRIX matrix_transpose(const MATRIX *mat);

/* In-place Arithmetic */
void matrix_add_inplace(MATRIX *a, const MATRIX *b);
void matrix_subtract_inplace(MATRIX *a, const MATRIX *b);
void matrix_scalar_multiply_inplace(MATRIX *mat, float scalar);

/* Element-wise Operations */
void matrix_apply(MATRIX *mat, float (*func)(float));

#endif