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

#ifndef __DENSE_H__
#define __DENSE_H__

#include "common.h"
#include "matrix.h"

typedef struct Dense {
    Matrix weights;
    Matrix bias;
} Dense;

int dense_init(Dense *layer, usize input_size, usize output_size);
void dense_destroy(Dense *layer);

Matrix dense_forward(Dense *layer, const Matrix *input);

#endif
