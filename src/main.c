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
#include "activation.h"
#include "dense.h"
#include "matrix.h"

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

    MNIST_IMAGE_HEADER images_header;
    MNIST_LABEL_HEADER labels_header;

    mnist_read_image_header(images, &images_header);
    mnist_read_label_header(labels, &labels_header); 

    float *images_data = malloc(images_header.count * sizeof(float));
    mnist_read_image_data_normalized(images, images_data, images_header.count);

    Matrix sample;
    matrix_create_buf(&sample, 1, 784, images_data);
    
    Dense layer;
    dense_init(&layer, 784, 128);
    Matrix layer_mat = dense_forward(&layer, &sample);
    matrix_apply(&layer_mat, ReLU);

    Dense output;
    dense_init(&output, 128, 10);
    Matrix output_mat = dense_forward(&output, &layer_mat);
    activation_Softmax(&output_mat);

    for (usize i = 0; i < 10; i++) {
        printf("%ld    ", i);
    }
    putchar('\n');

    for (usize i = 0; i < 10; i++) {
        printf("%.2f ", output_mat.data[i]);
    }
    putchar('\n');
    
    matrix_destroy(&sample);
    matrix_destroy(&layer_mat);
    matrix_destroy(&output_mat);
    
    dense_destroy(&layer);
    dense_destroy(&output);

    free(images_data);
    fclose(images);
    fclose(labels);

    return 0;
}
