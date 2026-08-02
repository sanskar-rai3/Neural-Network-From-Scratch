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
#include "dense.h"

int network_init(Network *network, const LayerConfig *config, usize layer_count) {
    network->layer = malloc(layer_count * sizeof(Layer));
    if (!network->layer) {
        return 0;
    }

    network->layer_count = layer_count;

    for (usize i = 0; i < layer_count; i++) {
        if (!dense_init(&network->layer[i].dense, config->input_size, config->output_size)) {
            while (i--) {
                dense_destroy(&network->layer[i].dense);

                free(network->layer);
                network->layer = NULL;
                network->layer_count = 0; 
            }

            return 0;
        }

        network->layer[i].activation = config[i].activation;
    }

    return 1;
}

void network_destroy(Network *network) {

}

Matrix network_forward(Network *network) {

}