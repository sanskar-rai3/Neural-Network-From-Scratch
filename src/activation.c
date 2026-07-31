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
#include "activation.h"

#include <math.h>

float ReLU(float val) {
    return val < 0 ? 0 : val;
}

float Leaky_ReLU(float val) {
    return val < 0 ? 0.01f * val : val;
}

float Tanh(float val) {
    return tanh(val);
}

float Sigmoid(float val) {
    return 1.0f / (1.0f + exp(-val));
}

float Softmax(float val) {

}
