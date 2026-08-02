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

/*==============================================================================
 * Header Structures
 *============================================================================*/

/**
 * @brief Represents the file header structure for MNIST image binary files.
 */
typedef struct MNIST_IMAGE_HEADER {
    int   magic; /**< Magic number identifier (0x00000803 / 2051 in big-endian) */
    usize count; /**< Total number of images in the dataset file */
    int   rows;  /**< Number of pixel rows per image (typically 28) */
    int   cols;  /**< Number of pixel columns per image (typically 28) */
} MNIST_IMAGE_HEADER;

/**
 * @brief Represents the file header structure for MNIST label binary files.
 */
typedef struct MNIST_LABEL_HEADER {
    int   magic; /**< Magic number identifier (0x00000801 / 2049 in big-endian) */
    usize count; /**< Total number of labels in the dataset file */
} MNIST_LABEL_HEADER;


/*==============================================================================
 * Reading Headers
 *============================================================================*/

/**
 * @brief Reads and parses the binary header from an MNIST image file.
 * 
 * Automatically handles big-endian to host-endian byte conversion.
 *
 * @param file   Pointer to the open binary file stream.
 * @param header Pointer to the MNIST_IMAGE_HEADER structure to populate.
 */
void mnist_read_image_header(FILE *file, MNIST_IMAGE_HEADER *header);

/**
 * @brief Reads and parses the binary header from an MNIST label file.
 * 
 * Automatically handles big-endian to host-endian byte conversion.
 *
 * @param file   Pointer to the open binary file stream.
 * @param header Pointer to the MNIST_LABEL_HEADER structure to populate.
 */
void mnist_read_label_header(FILE *file, MNIST_LABEL_HEADER *header);


/*==============================================================================
 * Reading Data
 *============================================================================*/

/**
 * @brief Reads raw image pixel bytes from an open MNIST image file.
 *
 * Reads (count * 28 * 28) raw byte values (0 to 255) into the target buffer.
 *
 * @param file  Pointer to the open binary file stream.
 * @param data  Pointer to the allocated byte buffer to fill.
 * @param count Number of images to read.
 */
void mnist_read_image_data(FILE *file, u8 *data, usize count);

/**
 * @brief Reads label byte values from an open MNIST label file.
 *
 * Reads 'count' raw byte values (0 to 9) into the target buffer.
 *
 * @param file  Pointer to the open binary file stream.
 * @param data  Pointer to the allocated byte buffer to fill.
 * @param count Number of labels to read.
 */
void mnist_read_label_data(FILE *file, u8 *data, usize count);

/**
 * @brief Reads image pixel bytes and normalizes them into float values in [0.0, 1.0].
 *
 * Reads pixel data and converts each pixel via (pixel_byte / 255.0f).
 *
 * @param file  Pointer to the open binary file stream.
 * @param data  Pointer to the allocated float buffer to fill (size: count * 784).
 * @param count Number of images to read and normalize.
 */
void mnist_read_image_data_normalized(FILE *file, float *data, usize count);

#endif