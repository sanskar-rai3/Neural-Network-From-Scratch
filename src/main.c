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
#include "layer.h"
#include "matrix.h"
#include "nn.h"
#include "render.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int main(void) {
    /* 1. Open MNIST Data Files */
    FILE *images = fopen("data/train-images-idx3-ubyte", "rb");
    if (!images) {
        perror("fopen images");
        return EXIT_FAILURE;
    }

    FILE *labels = fopen("data/train-labels-idx1-ubyte", "rb");
    if (!labels) {
        perror("fopen labels");
        fclose(images);
        return EXIT_FAILURE;
    }

    /* 2. Read Headers */
    MNIST_IMAGE_HEADER images_header;
    MNIST_LABEL_HEADER labels_header;

    mnist_read_image_header(images, &images_header);
    mnist_read_label_header(labels, &labels_header);

    /* 3. Load & Normalize Pixel Data (784 floats per image) */
    usize total_pixels = images_header.count * images_header.rows * images_header.cols;
    float *images_data = (float *)malloc(total_pixels * sizeof(float));
    if (!images_data) {
        fprintf(stderr, "Failed to allocate memory for images data\n");
        fclose(images);
        fclose(labels);
        return EXIT_FAILURE;
    }

    mnist_read_image_data_normalized(images, images_data, images_header.count);

    /* 4. Prepare Single Sample Input Matrix (1 x 784) */
    Matrix sample;
    matrix_copy_buffer(&sample, 1, 784, images_data);

    /* 5. Configure & Initialize Neural Network */
    LayerConfig lconfig[] = {
        { .input_size = 784, .output_size = 128, .activation = ACT_RELU },
        { .input_size = 128, .output_size = 10,  .activation = ACT_SOFTMAX }
    };
    usize layer_count = sizeof(lconfig) / sizeof(lconfig[0]);

    Network nn;
    if (!network_init(&nn, lconfig, layer_count)) {
        fprintf(stderr, "Failed to initialize neural network\n");
        matrix_destroy(&sample);
        free(images_data);
        fclose(images);
        fclose(labels);
        return EXIT_FAILURE;
    }

    /* 6. Perform Forward Pass */
    Matrix output;
    matrix_create(&output, 1, 10);

    network_forward(&output, &nn, &sample);

    /* 7. Display Results */
    printf("Class: ");
    for (usize i = 0; i < 10; i++) {
        printf("%zu\t", i);
    }
    printf("\nProb:  ");

    float sum = 0.0f;
    for (usize i = 0; i < 10; i++) {
        printf("%.2f\t", output.data[i]);
        sum += output.data[i];
    }
    printf("\nTotal Probability Sum: %.2f\n", sum);

    /* 8. Render the Sample */
    render_mnist_sample(&sample, "Sample"); 

    /* 9. Cleanup Allocations */
    matrix_destroy(&sample);
    matrix_destroy(&output);
    network_destroy(&nn);

    free(images_data);
    fclose(images);
    fclose(labels);

    return EXIT_SUCCESS;
}
