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

#ifndef __MNIST_H__
#define __MNIST_H__

#include "common.h"

#include <stdio.h>

/* Header Types */
typedef struct MNIST_IMAGE_HEADER {
    int   magic;
    usize count;
    int   rows;
    int   cols;
} MNIST_IMAGE_HEADER;

typedef struct MNIST_LABEL_HEADER {
    int   magic;
    usize count;
} MNIST_LABEL_HEADER;

/* Reading Headers */
void mnist_read_image_header(FILE *file, MNIST_IMAGE_HEADER *header);
void mnist_read_label_header(FILE *file, MNIST_LABEL_HEADER *header);

/* Reading Data */
void mnist_read_image_data(FILE *file, u8 *data, usize count);
void mnist_read_label_data(FILE *file, u8 *data, usize count);
void mnist_read_image_data_normalized(FILE *file, float *data, usize count);

#endif