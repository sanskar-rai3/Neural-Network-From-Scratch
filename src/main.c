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
#include "nn.h"
#include "layer.h"
#include "activation.h"
#include "optimizer.h"
#include "loss.h"
#include "matrix.h"

#include "mnist.h"
#include "render.h"

#include <stdio.h>
#include <stdlib.h>

#define EPOCH         5
#define LEARNING_RATE 0.01

#define INPUT_SIZE   784
#define OUTPUT_SIZE  10
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64

#define BATCH_SIZE 60

#define TRAIN_IMAGE_COUNT 60000
#define TRAIN_IMAGE_PATH  "data/train-images-idx3-ubyte"
#define TRAIN_LABEL_PATH  "data/train-labels-idx1-ubyte"

#define TEST_IMAGE_COUNT 10000
#define TEST_IMAGE_PATH  "data/t10k-images-idx3-ubyte"
#define TEST_LABEL_PATH  "data/t10k-labels-idx1-ubyte"

/* Helper functions */
void labels_to_one_hot(Matrix *target, const u8 *labels) {
    matrix_fill(target, 0.0f);

    for (usize i = 0; i < target->rows; i++) {
        target->data[i * target->cols + labels[i]] = 1.0f;
    }
}

int argmax(const Matrix *prediction) {
    int best = 0;

    for (int i = 1; i < (int)prediction->cols; i++) {
        if (prediction->data[i] > prediction->data[best]) {
            best = i;
        }
    }

    return best;
}

int main(void) {
    /* Opening mnist train files */
    FILE *train_image = fopen(TRAIN_IMAGE_PATH, "rb");
    if (!train_image) {
        perror("train_image Error");
        return 1;
    }

    FILE *train_label = fopen(TRAIN_LABEL_PATH, "rb");
    if (!train_label) {
        perror("train_label Error");
        fclose(train_image);
        return 1;
    }

    /* Reading mnist train headers */
    MNIST_IMAGE_HEADER image_header;
    MNIST_LABEL_HEADER label_header;

    mnist_read_image_header(train_image, &image_header);
    mnist_read_label_header(train_label, &label_header);

    if (image_header.count != label_header.count) {
        fprintf(stderr, "Image/label count mismatch in training data\n");

        fclose(train_image);
        fclose(train_label);

        return 1;
    }

    /* Allocating memory for mnist train data */
    float *train_image_data = malloc(TRAIN_IMAGE_COUNT * INPUT_SIZE * sizeof(float));
    if (!train_image_data) {
        perror("train_image_data");

        fclose(train_image);
        fclose(train_label);

        return 1;
    }

    u8 *train_label_data = malloc(TRAIN_IMAGE_COUNT * sizeof(u8));
    if (!train_label_data) {
        perror("train_label_data");

        fclose(train_image);
        fclose(train_label);

        return 1;
    }

    /* Reading mnist train data */
    mnist_read_image_data_normalized(train_image, train_image_data, TRAIN_IMAGE_COUNT);   
    mnist_read_label_data(train_label, train_label_data, TRAIN_IMAGE_COUNT);

    /* Closing mnist train files */
    fclose(train_image);
    fclose(train_label);

    /*=====================================================================*/

    /* Configuring layers */
    LayerConfig configs[] = {
        {
            .input_size  = INPUT_SIZE,
            .output_size = HIDDEN1_SIZE,
            .activation  = ACT_RELU
        },
        {
            .input_size  = HIDDEN1_SIZE,
            .output_size = HIDDEN2_SIZE,
            .activation  = ACT_RELU
        },
        {
            .input_size  = HIDDEN2_SIZE,
            .output_size = OUTPUT_SIZE,
            .activation  = ACT_SOFTMAX_CEL
        }
    };
    
    /* Network initialization */
    Network nn;
    if (!network_init(&nn, configs, sizeof(configs) / sizeof(configs[0]))) {
        fprintf(stderr, "Error initializing network");
        
        free(train_image_data);
        free(train_label_data);

        return 1;
    }

    /* Optimizer initialization */
    Optimizer optimizer;
    optimizer_init(&optimizer, LEARNING_RATE, OPT_SGD);

    /*=====================================================================*/

    /* Creating training matrices */
    Matrix X, Y, prediction, d_output;

    if (!matrix_create(&X, BATCH_SIZE, INPUT_SIZE) ||
        !matrix_create(&Y, BATCH_SIZE, OUTPUT_SIZE) ||
        !matrix_create(&prediction, BATCH_SIZE, OUTPUT_SIZE) ||
        !matrix_create(&d_output, BATCH_SIZE, OUTPUT_SIZE)) {

        fprintf(stderr, "Failed to allocate training matrices\n");

        matrix_destroy(&X);
        matrix_destroy(&Y);
        matrix_destroy(&prediction);
        matrix_destroy(&d_output);

        optimizer_destroy(&optimizer);
        network_destroy(&nn);

        free(train_image_data);
        free(train_label_data);

        return 1;
    }

    /* Training */
    for (usize epoch = 0; epoch < EPOCH; epoch++) {
        float epoch_loss = 0.0f;

        for (usize batch = 0; batch < TRAIN_IMAGE_COUNT / BATCH_SIZE; batch++) {
            usize start = batch * BATCH_SIZE;

            /* Fill batch input */
            for (usize i = 0; i < BATCH_SIZE; i++) {
                for (usize j = 0; j < INPUT_SIZE; j++) {
                    X.data[i * INPUT_SIZE + j] =
                        train_image_data[
                            (start + i) * INPUT_SIZE + j
                        ];
                }
            }

            /* Convert label to one hot */
            labels_to_one_hot(&Y, train_label_data + start);

            /* Forward pass */
            network_forward(&prediction, &nn, &X);

            /* Calculate loss */
            float loss = loss_cross_entropy(&prediction, &Y);
            epoch_loss += loss;

            /*=================================================================
             * Softmax + Cross-Entropy backward
             *
             * dZ = prediction - target
             ================================================================*/
            loss_cross_entropy_softmax_backward(&d_output, &prediction, &Y);

            /* Back propagation */
            network_backward(&nn, &d_output);

            /* Update weights and biases */
            optimizer_step(&optimizer, &nn);
        }

        epoch_loss /= (float)(TRAIN_IMAGE_COUNT / (1.0f * BATCH_SIZE));

        printf("epoch: %2zu | loss: %.6f\n", epoch + 1, epoch_loss);
    }

    /*=====================================================================*/

    /* Opening mnist test files */
    FILE *test_image = fopen(TEST_IMAGE_PATH, "rb");
    if (!test_image) {
        perror("test_image");
        return 1;
    }

    FILE *test_label = fopen(TEST_LABEL_PATH, "rb");
    if (!test_label) {
        perror("test_label");
        fclose(test_image);
        return 1;
    }

    /* Reading mnist test headers */
    MNIST_IMAGE_HEADER test_image_header;
    MNIST_LABEL_HEADER test_label_header;

    mnist_read_image_header(test_image, &test_image_header);
    mnist_read_label_header(test_label, &test_label_header);

    if (test_image_header.count != test_label_header.count) {
        fprintf(stderr, "Test image/label count mismatch\n");

        fclose(test_image);
        fclose(test_label);

        return 1;
    }

    /* Allocating memory for mnist test data */
    float *test_image_data = malloc(TEST_IMAGE_COUNT * INPUT_SIZE * sizeof(float));
    if (!test_image_data) {
        perror("test_image_data");

        fclose(test_image);
        fclose(test_label);

        return 1;
    }

    u8 *test_label_data= malloc(TEST_IMAGE_COUNT * sizeof(u8));
    if (!test_label_data) {
        perror("test_label_data");

        fclose(test_image);
        fclose(test_label);

        return 1;
    }

    /* Reading mnist test data */
    mnist_read_image_data_normalized(test_image, test_image_data, TEST_IMAGE_COUNT);
    mnist_read_label_data(test_label, test_label_data, TEST_IMAGE_COUNT);

    /* Closing mnist test files */
    fclose(test_image);
    fclose(test_label);

    /*=====================================================================*/

    /* Testing multiple samples */
    for (usize i = 0; i < 10; i++){
        /* Creating input sample */
        Matrix pred_input;
        if (!matrix_copy_buffer(&pred_input, 1, INPUT_SIZE, test_image_data + 784 * i)) {
            fprintf(stderr, "Failed to create input matrix\n");

            optimizer_destroy(&optimizer);
            network_destroy(&nn);

            free(train_image_data);
            free(train_label_data);
            free(test_image_data);
            free(test_label_data);

            return 1;
        }

        /* Prediction output */
        Matrix pred_output;

        if (!matrix_create(&pred_output, 1, OUTPUT_SIZE)) {
            fprintf(stderr, "Failed to create prediction matrix\n");

            matrix_destroy(&X);
            optimizer_destroy(&optimizer);
            network_destroy(&nn);

            free(train_image_data);
            free(train_label_data);
            free(test_image_data);
            free(test_label_data);

            return 1;
        }

        /* Forward pass */
        network_forward(&pred_output, &nn, &pred_input);

        /* Print all the values of output neuron */
        for (usize i = 0; i < 10; i++) {
            printf("%f |", pred_output.data[i]);
        }
        putchar('\n');

        /* Get predicted class */
        usize predicted = argmax(&pred_output);

        printf("\n==========Prediction==========\n");
        printf("Actual:     %u\n", test_label_data[i]);
        printf("Predicted:  %zu\n", predicted);
        printf("Confidence: %.2f%%\n", pred_output.data[predicted] * 100.0f);

        render_mnist_sample(&pred_input, "Sample");

        matrix_destroy(&pred_input);
        matrix_destroy(&pred_output);
    }

    /* Clean up */
    matrix_destroy(&X);
    matrix_destroy(&Y);
    matrix_destroy(&prediction);
    matrix_destroy(&d_output);

    optimizer_destroy(&optimizer);
    network_destroy(&nn);

    free(train_image_data);
    free(train_label_data);
    free(test_image_data);
    free(test_label_data);

    return 0;
}
