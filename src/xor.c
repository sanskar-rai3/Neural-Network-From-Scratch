#include "common.h"
#include "nn.h"
#include "layer.h"
#include "activation.h"
#include "optimizer.h"
#include "loss.h"
#include "matrix.h"

#include <stdio.h>

#define EPOCH         10000
#define LEARNING_RATE 0.01

#define INPUT_SIZE  2
#define HIDDEN1_SIZE 16 
#define OUTPUT_SIZE 1

#define TRAIN_INPUT_COUNT 4

float train_x_data[8] = {
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    1.0f, 1.0f
};

float train_label[4] = {
    0.0f,
    1.0f,
    1.0f,
    0.0f
};

float test_x_data[8] = {
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 0.0f,
    1.0f, 1.0f
};

float test_label[4] = {
    0.0f,
    1.0f,
    1.0f,
    0.0f
};

int main(void) {
    LayerConfig configs[] = {
        {
            .input_size  = INPUT_SIZE,
            .output_size = HIDDEN1_SIZE,
            .activation  = ACT_SIGMOID
        },

        {
            .input_size  = HIDDEN1_SIZE,
            .output_size = OUTPUT_SIZE,
            .activation  = ACT_SIGMOID
        }
    };
    
    /* Network initialization */
    Network nn;
    if (!network_init(&nn, configs, sizeof(configs) / sizeof(configs[0]))) {
        fprintf(stderr, "Error initializing network");
        
        return 1;
    }

    /* Optimizer initialization */
    Optimizer optimizer;
    optimizer_init(&optimizer, LEARNING_RATE, OPT_SGD);

    /*=====================================================================*/

    /* Creating training matrices */
    Matrix X, Y, prediction, d_output;

    if (!matrix_create(&X, 1, INPUT_SIZE) ||
        !matrix_create(&Y, 1, OUTPUT_SIZE) ||
        !matrix_create(&prediction, 1, OUTPUT_SIZE) ||
        !matrix_create(&d_output, 1, OUTPUT_SIZE)) {

        fprintf(stderr, "Failed to allocate training matrices\n");

        matrix_destroy(&X);
        matrix_destroy(&Y);
        matrix_destroy(&prediction);
        matrix_destroy(&d_output);

        optimizer_destroy(&optimizer);
        network_destroy(&nn);

        return 1;
    }

    /* Training */
    for (usize epoch = 0; epoch < EPOCH; epoch++) {
        float epoch_loss = 0.0f;

        for (usize sample = 0; sample < TRAIN_INPUT_COUNT; sample++) {
            for (usize i = 0; i < INPUT_SIZE; i++) {
                X.data[i] = train_x_data[sample * INPUT_SIZE + i];
            }

            Y.data[0] = train_label[sample];

            /* Forward pass */
            network_forward(&prediction, &nn, &X);

            /* Calculate loss */
            float loss = loss_mse(&prediction, &Y);
            epoch_loss += loss;

            loss_mse_backward(&d_output, &prediction, &Y);

            /* Back propagation */
            network_backward(&nn, &d_output);

            /* Update weights and biases */
            optimizer_step(&optimizer, &nn);
        }

        epoch_loss /= (float)TRAIN_INPUT_COUNT;

        printf("epoch: %2zu | loss: %.6f\n", epoch + 1, epoch_loss);
    }

    for (usize i = 0; i < 4; i++){
        Matrix pred_input;
        if (!matrix_copy_buffer(&pred_input, 1, INPUT_SIZE, test_x_data + 2 * i)) {
            fprintf(stderr, "Failed to create input matrix\n");

            optimizer_destroy(&optimizer);
            network_destroy(&nn);

            return 1;
        }

        /* Prediction output */
        Matrix pred_output;

        if (!matrix_create(&pred_output, 1, OUTPUT_SIZE)) {
            fprintf(stderr, "Failed to create prediction matrix\n");

            matrix_destroy(&X);
            optimizer_destroy(&optimizer);
            network_destroy(&nn);

            return 1;
        }

        /* Forward pass */
        network_forward(&pred_output, &nn, &pred_input);

        printf("\n==========Prediction==========\n");
        printf("Actual:     %f\n", test_label[i]);
        printf("Predicted:  %f\n", pred_output.data[0]);

        matrix_destroy(&pred_input);
        matrix_destroy(&pred_output);
    }

    matrix_destroy(&X);
    matrix_destroy(&Y);
    matrix_destroy(&prediction);
    matrix_destroy(&d_output);

    optimizer_destroy(&optimizer);
    network_destroy(&nn);

    return 0;
}
