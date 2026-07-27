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
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

static inline u32 bswap32(u32 byte) {
#if (defined(__GNUC__) || defined(__clang__))
    return __builtin_bswap32(byte);
#else
    return (
        (byte & 0x000000FFu) << 24  |
        (byte & 0x0000FF00u) << 8   |
        (byte & 0x00FF0000u) >> 8   |
        (byte & 0xFF000000u) >> 24     
    );
#endif
}

static u32 read_be(FILE *file) {
    u32 read;
    fread(&read, sizeof(read), 1, file);

    return bswap32(read);
}

void mnist_read_images_header(FILE *file, MNIST_IMAGES_HEADER *header) {
    header->magic = read_be(file);
    header->count = read_be(file);

    header->rows  = read_be(file);
    header->cols  = read_be(file);
    if (header->rows != 28 || header->cols != 28) {
        fprintf(stderr, "Unsupported image size.\n");
        exit(EXIT_FAILURE);
    }
}

void mnist_read_labels_header(FILE *file, MNIST_LABELS_HEADER *header) {
    header->magic = read_be(file);
    header->count = read_be(file);
}

void mnist_read_images_data(FILE *file, u8 *data, size_t count) {
    fread(data, sizeof(u8), count, file);
}

void mnist_read_labels_data(FILE *file, u8 *data, size_t count) {
    fread(data, sizeof(u8), count, file);
}

void mnist_read_images_data_normalized(FILE *file, float *datas, size_t count) {
    u8 *pixels = malloc(count * sizeof(u8));
    if (!pixels) {
        perror("malloc");
        return;
    }

    mnist_read_images_data(file, pixels, count);

    for (size_t i = 0; i < count; i++) {
        datas[i] = pixels[i] / 255.0f; 
    }

    free(pixels);
}