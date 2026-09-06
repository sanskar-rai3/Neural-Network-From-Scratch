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
#include "emnist_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*==============================================================================
 * Helper Functions
 *============================================================================*/

static inline u32 bswap32(u32 val) {
#if (defined(__GNUC__) || defined(__clang__))
    return __builtin_bswap32(val);
#else
    return ((val & 0x000000FFu) << 24) |
           ((val & 0x0000FF00u) << 8)  |
           ((val & 0x00FF0000u) >> 8)  |
           ((val & 0xFF000000u) >> 24);
#endif
}

static u32 read_be(FILE *file) {
    assert(file);

    u32 val = 0;
    size_t bytes_read = fread(&val, sizeof(val), 1, file);
    if (bytes_read){} /* To remove compiler warning */

    assert(bytes_read == 1);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return bswap32(val);
#else
    return val;
#endif
}

/*==============================================================================
 * Reading Headers
 *============================================================================*/

void emnist_read_image_header(FILE *file, EMNIST_IMAGE_HEADER *header) {
    if (!file || !header) {
        fprintf(stderr, "Error: Invalid argument passed to mnist_read_image_header\n");
        exit(EXIT_FAILURE);
    }

    header->magic = read_be(file);
    if (header->magic != 2051) {
        fprintf(stderr, "Error: Invalid image file magic number (%d)\n", header->magic);
        exit(EXIT_FAILURE);
    }

    header->count = read_be(file);

    header->rows = read_be(file);
    header->cols = read_be(file);
    if (header->rows != 28 || header->cols != 28) {
        fprintf(stderr, "Error: Unsupported image dimensions (%dx%d)\n", header->rows, header->cols);
        exit(EXIT_FAILURE);
    }
}

void emnist_read_label_header(FILE *file, EMNIST_LABEL_HEADER *header) {
    if (!file || !header) {
        fprintf(stderr, "Error: Invalid argument passed to mnist_read_label_header\n");
        exit(EXIT_FAILURE);
    }

    header->magic = read_be(file);
    if (header->magic != 2049) {
        fprintf(stderr, "Error: Invalid label file magic number (%d)\n", header->magic);
        exit(EXIT_FAILURE);
    }

    header->count = read_be(file);
}

/*==============================================================================
 * Reading Data
 *============================================================================*/

void emnist_read_image_data(FILE *file, u8 *data, usize count) {
    if (!file || !data || count == 0) {
        fprintf(stderr, "Error: Invalid argument passed to mnist_read_image_data\n");
        exit(EXIT_FAILURE);
    }

    /* Each image contains 28 * 28 = 784 byte pixels */
    usize total_pixels = count * 28 * 28;
    size_t bytes_read = fread(data, sizeof(u8), total_pixels, file);
    if (bytes_read != total_pixels) {
        fprintf(stderr, "Error: Failed to read full image data\n");
        exit(EXIT_FAILURE);
    }
}

void emnist_read_label_data(FILE *file, u8 *data, usize count) {
    if (!file || !data || count == 0) {
        fprintf(stderr, "Error: Invalid argument passed to mnist_read_label_data\n");
        exit(EXIT_FAILURE);
    }

    size_t bytes_read = fread(data, sizeof(u8), count, file);
    if (bytes_read != count) {
        fprintf(stderr, "Error: Failed to read full label data\n");
        exit(EXIT_FAILURE);
    }
}

void emnist_read_image_data_normalized(FILE *file, float *data, usize count) {
    if (!file || !data || count == 0) {
        fprintf(stderr, "Error: Invalid argument passed to mnist_read_image_data_normalized\n");
        exit(EXIT_FAILURE);
    }

    usize total_pixels = count * 28 * 28;
    u8 *pixels = (u8 *)malloc(total_pixels * sizeof(u8));
    if (!pixels) {
        fprintf(stderr, "Error: Failed to allocate pixel staging buffer\n");
        exit(EXIT_FAILURE);
    }

    emnist_read_image_data(file, pixels, count);

    for (usize i = 0; i < total_pixels; i++) {
        data[i] = (float)pixels[i] / 255.0f;
    }

    free(pixels);
}
