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
#include "optimizer.h"
#include "loss.h"
#include "math/matrix.h"

#include "data-loader/emnist_loader.h"

#include <stdio.h>
#include <stdlib.h>

#define EPOCH         5
#define LEARNING_RATE 0.01

#define INPUT_SIZE  (EMNIST_ROWS * EMNIST_COLS)
#define HIDDEN1_SIZE 256
#define HIDDEN2_SIZE 128
#define OUTPUT_SIZE  10

#define BATCH_SIZE 1

#define TRAIN_IMAGE_PATH  "data/mnist/emnist-mnist-train-images-idx3-ubyte"
#define TRAIN_LABEL_PATH  "data/mnist/emnist-mnist-train-labels-idx1-ubyte"

#define TEST_IMAGE_PATH  "data/mnist/emnist-mnist-test-images-idx3-ubyte"
#define TEST_LABEL_PATH  "data/mnist/emnist-mnist-test-labels-idx1-ubyte"

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
    EMNIST_IMAGE_HEADER train_image_header;
    EMNIST_LABEL_HEADER train_label_header;
    
    emnist_read_image_header(&train_image_header, train_image);
    emnist_read_label_header(&train_label_header, train_label);
    
    const usize TRAIN_IMAGE_COUNT = train_image_header.count;
    
    if (train_image_header.count != train_label_header.count) {
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
    emnist_read_image_data_normalized(train_image_data, train_image, TRAIN_IMAGE_COUNT);   
    emnist_read_label_data(train_label_data, train_label, TRAIN_IMAGE_COUNT);
    
    /* Closing mnist train files */
    fclose(train_image);
    fclose(train_label);
    
    /*=====================================================================*/

    /* Configuring layers */
    // LayerConfig configs[] = {
    //     {
    //         .layer_type  = LAYER_DENSE,
    //
    //         .dense_config.input_size  = INPUT_SIZE,
    //         .dense_config.output_size = HIDDEN1_SIZE,
    //         
    //         .activation = ACT_RELU
    //     },
    //     {
    //         .layer_type = LAYER_DENSE,
    //
    //         .dense_config.input_size  = HIDDEN1_SIZE,
    //         .dense_config.output_size = HIDDEN2_SIZE,
    //
    //         .activation = ACT_RELU
    //     },
    //     {
    //         .layer_type = LAYER_DENSE,
    //
    //         .dense_config.input_size  = HIDDEN2_SIZE,
    //         .dense_config.output_size = OUTPUT_SIZE,
    //
    //         .activation = ACT_SOFTMAX_CEL
    //     }
    // };
    
    /* Network initialization */
    Network nn;
    // if (!network_init(&nn, configs, sizeof(configs) / sizeof(configs[0]))) {
    //     fprintf(stderr, "Error initializing network");
    //     
    //     free(train_image_data);
    //     free(train_label_data);
    //
    //     return 1;
    // }

    network_load(&nn, "models/mnist-model-10epoch.nmf");

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

    network_save(&nn, "models/mnist-model-15epoch.nmf");

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
    EMNIST_IMAGE_HEADER test_image_header;
    EMNIST_LABEL_HEADER test_label_header;

    emnist_read_image_header(&test_image_header, test_image);
    emnist_read_label_header(&test_label_header, test_label);

    const usize TEST_IMAGE_COUNT = test_image_header.count;

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
    emnist_read_image_data_normalized(test_image_data, test_image, TEST_IMAGE_COUNT);
    emnist_read_label_data(test_label_data, test_label, TEST_IMAGE_COUNT);

    /* Closing mnist test files */
    fclose(test_image);
    fclose(test_label);

    /*=====================================================================*/

    /* Testing Model Accuracy */
    usize correct = 0;

    Matrix pred_input;
    Matrix pred_output;

    if (!matrix_create(&pred_input, 1, INPUT_SIZE) ||
        !matrix_create(&pred_output, 1, OUTPUT_SIZE)) {

        fprintf(stderr, "Failed to allocate test matrices\n");

        matrix_destroy(&pred_input);
        matrix_destroy(&pred_output);

        optimizer_destroy(&optimizer);
        network_destroy(&nn);

        free(train_image_data);
        free(train_label_data);
        free(test_image_data);
        free(test_label_data);

        return 1;
    }

    for (usize i = 0; i < TEST_IMAGE_COUNT; i++) {
        /* Copy one image into input */
        for (usize j = 0; j < INPUT_SIZE; j++) {
            pred_input.data[j] = test_image_data[i * INPUT_SIZE + j];
        }

        /* Forward pass */
        network_forward(&pred_output, &nn, &pred_input);

        /* Get predicted class */
        usize predicted = argmax(&pred_output);

        /* Compare */
        if (predicted == test_label_data[i]) {
            correct++;
        }
    }

    float accuracy = (float)correct / (float)TEST_IMAGE_COUNT * 100.0f;

    printf("\n========== Test Accuracy ==========\n");
    printf("Correct:   %zu / %zu\n", correct, TEST_IMAGE_COUNT);
    printf("Accuracy:  %.2f%%\n", accuracy);

    matrix_destroy(&pred_input);
    matrix_destroy(&pred_output);

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
