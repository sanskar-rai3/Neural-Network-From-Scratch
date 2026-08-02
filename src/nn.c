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

#include <stdlib.h>
#include <assert.h>

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

int network_init(Network *network, const LayerConfig *configs, usize layer_count) {
    assert(network != NULL);
    assert(configs != NULL);
    assert(layer_count > 0);

    network->layers = (Layer *)malloc(layer_count * sizeof(Layer));
    if (!network->layers) {
        return 0;
    }

    network->layer_count = layer_count;

    for (usize i = 0; i < layer_count; i++) {
        if (!layer_init(&network->layers[i], &configs[i])) {
            while (i > 0) {
                i--;
                layer_destroy(&network->layers[i]);
            }
            
            free(network->layers);
            network->layers = NULL;
            network->layer_count = 0;

            return 0;
        }
    }

    return 1;
}

void network_destroy(Network *network) {
    assert(network != NULL);

    if (network->layers != NULL) {
        for (usize i = 0; i < network->layer_count; i++) {
            layer_destroy(&network->layers[i]);
        }

        free(network->layers);
        network->layers = NULL;
    }

    network->layer_count = 0;
}

/*==============================================================================
 * Forward Pass
 *============================================================================*/

void network_forward(Matrix *output, const Network *network, const Matrix *input) {
    assert(output != NULL);
    assert(network != NULL);
    assert(input != NULL);
    assert(network->layer_count > 0);

    /* Layer 1: Run input directly into a temp matrix */
    Matrix current;
    matrix_create(&current, input->rows, network->layers[0].dense.weights.cols);
    layer_forward(&current, &network->layers[0], input);

    /* Middle Layers: Run current -> next, swap */
    for (usize i = 1; i < network->layer_count - 1; i++) {
        Matrix next;
        matrix_create(&next, input->rows, network->layers[i].dense.weights.cols);

        layer_forward(&next, &network->layers[i], &current);

        matrix_destroy(&current);
        current = next;
    }

    /* Final Layer: Compute directly into output */
    if (network->layer_count > 1) {
        layer_forward(output, &network->layers[network->layer_count - 1], &current);
        matrix_destroy(&current);
    } else {
        /* If 1 layer total, output gets current's data */
        *output = current;
    }
}