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
#include "matrix.h"
#include "layer.h"
#include "nn.h"
#include "loss.h"
#include "optimizer.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    /*======================================================================
     * XOR Dataset
     *======================================================================*/

    float x_data[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };

    float y_data[] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    Matrix X = matrix_empty();
    Matrix Y = matrix_empty();

    if (!matrix_copy_buffer(&X, 4, 2, x_data)) {
        fprintf(stderr, "Failed to create X\n");
        return EXIT_FAILURE;
    }

    if (!matrix_copy_buffer(&Y, 4, 1, y_data)) {
        fprintf(stderr, "Failed to create Y\n");
        matrix_destroy(&X);
        return EXIT_FAILURE;
    }

    /*======================================================================
     * Network Architecture
     *
     * 2 inputs
     *    ↓
     * 4 hidden neurons + sigmoid
     *    ↓
     * 1 output neuron + sigmoid
     *======================================================================*/

    LayerConfig configs[] = {
        {
            .input_size  = 2,
            .output_size = 4,
            .activation  = ACT_SIGMOID
        },
        {
            .input_size  = 4,
            .output_size = 1,
            .activation  = ACT_SIGMOID
        }
    };

    Network network;

    if (!network_init(&network, configs, 2)) {
        fprintf(stderr, "Failed to initialize network\n");
        matrix_destroy(&X);
        matrix_destroy(&Y);
        return EXIT_FAILURE;
    }

    /*======================================================================
     * Optimizer
     *======================================================================*/

    Optimizer optimizer;

    optimizer_init(
        &optimizer,
        0.5f,
        OPT_SGD
    );

    /*======================================================================
     * Training Buffers
     *======================================================================*/

    Matrix prediction = matrix_empty();
    Matrix d_output   = matrix_empty();

    if (!matrix_create(&prediction, 4, 1)) {
        fprintf(stderr, "Failed to create prediction matrix\n");

        optimizer_destroy(&optimizer);
        network_destroy(&network);
        matrix_destroy(&X);
        matrix_destroy(&Y);

        return EXIT_FAILURE;
    }

    if (!matrix_create(&d_output, 4, 1)) {
        fprintf(stderr, "Failed to create gradient matrix\n");

        matrix_destroy(&prediction);
        optimizer_destroy(&optimizer);
        network_destroy(&network);
        matrix_destroy(&X);
        matrix_destroy(&Y);

        return EXIT_FAILURE;
    }

    /*======================================================================
     * Training
     *======================================================================*/

    const usize epochs = 10000;

    for (usize epoch = 0; epoch < epochs; epoch++) {

        /* Forward:
         *
         * X → Layer 0 → Layer 1 → prediction
         */
        network_forward(
            &prediction,
            &network,
            &X
        );

        /* Compute loss */
        float loss = loss_mse(
            &prediction,
            &Y
        );

        /* Print progress */
        if (epoch % 1000 == 0) {
            printf(
                "epoch: %5zu | loss: %.6f\n",
                epoch,
                loss
            );
        }

        /*==================================================================
         * dL/dA
         *
         * MSE:
         *
         * dL/dA = 2(A - Y) / N
         *==================================================================*/

        loss_mse_backward(
            &d_output,
            &prediction,
            &Y
        );

        /*==================================================================
         * Backpropagation
         *
         * d_output enters the final layer as dL/dA.
         *
         * network_backward propagates:
         *
         * dL/dA
         *   ↓
         * Layer 1
         *   ↓
         * Layer 0
         *==================================================================*/

        network_backward(
            &network,
            &d_output
        );

        /*==================================================================
         * Update weights and biases
         *==================================================================*/

        optimizer_step(
            &optimizer,
            &network
        );
    }

    /*======================================================================
     * Final Predictions
     *======================================================================*/

    network_forward(
        &prediction,
        &network,
        &X
    );

    printf("\nXOR predictions:\n");

    for (usize i = 0; i < 4; i++) {
        printf(
            "%.0f XOR %.0f = %.6f\n",
            X.data[i * 2],
            X.data[i * 2 + 1],
            prediction.data[i]
        );
    }

    /*======================================================================
     * Cleanup
     *======================================================================*/

    matrix_destroy(&prediction);
    matrix_destroy(&d_output);

    optimizer_destroy(&optimizer);
    network_destroy(&network);

    matrix_destroy(&X);
    matrix_destroy(&Y);

    return EXIT_SUCCESS;
}
