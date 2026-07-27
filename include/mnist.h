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

#include <stdio.h>
#include <stddef.h>

/* Header Types */
typedef struct MNIST_IMAGES_HEADER {
    int    magic;
    size_t count;
    int    rows;
    int    cols;
} MNIST_IMAGES_HEADER;

typedef struct MNIST_LABELS_HEADER {
    int    magic;
    size_t count;
} MNIST_LABELS_HEADER;

/* Reading Headers */
void mnist_read_images_header(FILE *file, MNIST_IMAGES_HEADER *header);
void mnist_read_labels_header(FILE *file, MNIST_LABELS_HEADER *header);

/* Reading Contents */
void mnist_read_images_data(FILE *file, u8 *data, size_t count);
void mnist_read_labels_data(FILE *file, u8 *data, size_t count);
void mnist_read_images_data_normallized(FILE *file, float *datas, size_t count);