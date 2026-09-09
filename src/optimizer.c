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

#include "optimizer.h"
#include "layer/dense.h"
#include "nn.h"

#include <assert.h>

/*==============================================================================
 * Creation & Destruction
 *============================================================================*/

void optimizer_init(Optimizer *optimizer, float learning_rate, OptimizerType type) {
    assert(optimizer);
    assert(learning_rate > 0.0f);

    optimizer->type = type;
    optimizer->learning_rate = learning_rate;
}

void optimizer_destroy(Optimizer *optimizer) {
    assert(optimizer);

    optimizer->type = OPT_NULL;
    optimizer->learning_rate = 0.0f;
}

/*==============================================================================
 * Optimization
 *============================================================================*/

void optimizer_step(Optimizer *optimizer, Network *network) {
    assert(optimizer);
    assert(network);

    switch (optimizer->type) {

        case OPT_SGD:
            for (usize i = 0; i < network->layer_count; i++) {
                Dense *dense = &network->layers[i].dense;

                /*
                 * W = W - learning_rate * dW
                 */
                matrix_scalar_multiply_inplace(
                    &dense->d_weights,
                    optimizer->learning_rate
                );

                matrix_subtract_inplace(
                    &dense->weights,
                    &dense->d_weights
                );

                /*
                 * b = b - learning_rate * db
                 */
                matrix_scalar_multiply_inplace(
                    &dense->d_bias,
                    optimizer->learning_rate
                );

                matrix_subtract_inplace(
                    &dense->bias,
                    &dense->d_bias
                );
            }
            break;

        case OPT_NULL:
        default:
            break;
    }
}
