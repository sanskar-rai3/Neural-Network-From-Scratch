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

/* Helper Funtions */
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
    if (fread(&read, sizeof(read), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return bswap32(read);
#else
    return read;
#endif
}

/* Reading Headers */
void mnist_read_image_header(FILE *file, MNIST_IMAGE_HEADER *header) {
    header->magic = read_be(file);
    if (header->magic != 2051) {
        fprintf(stderr, "Invalid image file.\n");
        exit(EXIT_FAILURE);
    }

    header->count = read_be(file);

    header->rows  = read_be(file);
    header->cols  = read_be(file);
    if (header->rows != 28 || header->cols != 28) {
        fprintf(stderr, "Unsupported image size\n");
        exit(EXIT_FAILURE);
    }
}

void mnist_read_label_header(FILE *file, MNIST_LABEL_HEADER *header) {
    header->magic = read_be(file);
    if (header->magic != 2049) {
        fprintf(stderr, "Invalid label file\n");
        exit(EXIT_FAILURE);
    }

    header->count = read_be(file);
}

/* Reading Datas */
void mnist_read_image_data(FILE *file, u8 *data, usize count) {
    if (fread(data, sizeof(u8), count, file) != count) {
        fprintf(stderr, "Failed to read image data\n");
        exit(EXIT_FAILURE);
    }
}

void mnist_read_label_data(FILE *file, u8 *data, usize count) {
    if (fread(data, sizeof(u8), count, file) != count) {
        fprintf(stderr, "Failed to read label data\n");
        exit(EXIT_FAILURE);
    }
}

void mnist_read_image_data_normalized(FILE *file, float *data, usize count) {
    u8 *pixels = malloc(count * sizeof(u8));
    if (!pixels) {
        perror("malloc");
        return;
    }

    mnist_read_image_data(file, pixels, count);

    for (size_t i = 0; i < count; i++) {
        data[i] = pixels[i] / 255.0f; 
    }

    free(pixels);
}
