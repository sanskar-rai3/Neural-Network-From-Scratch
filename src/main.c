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

#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int main() {
    FILE *images = fopen("data/train-images-idx3-ubyte", "rb");
    if (!images) {
        perror("fopen images");
        return EXIT_FAILURE;
    }

    FILE *labels = fopen("data/train-labels-idx1-ubyte", "rb");
    if (!labels) {
        perror("fopen labels");
        return EXIT_FAILURE;
    }

    MNIST_IMAGES_HEADER images_header;
    MNIST_LABELS_HEADER labels_header;

    mnist_read_images_header(images, &images_header);
    mnist_read_labels_header(labels, &labels_header); 

    // printf("Image headers\n");
    // printf("Magic: %" PRIu32 "\n", images_header.magic);
    // printf("Image Count: %" PRIu32 "\n", images_header.count);
    // printf("Rows: %" PRIu32 "\n", images_header.rows);
    // printf("Cols: %" PRIu32 "\n", images_header.cols);

    // printf("\nLabels header\n");
    // printf("Magic: %" PRIu32 "\n", labels_header.magic);
    // printf("Label Count: %" PRIu32 "\n", labels_header.count);
    
    u8 *labels_data = malloc(labels_header.count * sizeof(u8));
    mnist_read_labels_data(labels, labels_data, labels_header.count);

    u8 *images_data = malloc(images_header.count * sizeof(u8));
    mnist_read_images_data(images, images_data, images_header.count);
    
    // for (int i = 0; i < labels_header.count; i++) {
    //     printf("%" PRIu8 " ", labels_data[i]);
    // }
    

    free(labels_data);
    free(images_data);
    fclose(images);
    fclose(labels);

    return 0;
}