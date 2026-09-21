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

#ifndef TENSOR_H
#define TENSOR_H

#include "common.h"

typedef struct {
    usize *shape;
    usize rank;

    usize *strides;

    usize size;
    float *data;
} Tensor;

void tensor_create(Tensor *t, usize rank, const usize *shape);
void tensor_destroy(Tensor *t);

void tensor_copy(Tensor *dest, const Tensor *src);

void tensor_fill(Tensor *t, float val);

void tensor_flatten(Tensor *t);

#endif /* TENSOR_H */