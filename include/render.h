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

#ifndef __RENDER_H_
#define __RENDER_H_

#include "common.h"
#include "matrix.h"

#define IMAGE_SIZE 28
#define SCALE_FACTOR 15
#define WINDOW_SIZE (IMAGE_SIZE * SCALE_FACTOR)

/**
 * @brief Displays a single 28x28 MNIST matrix sample in an SDL2 window.
 * 
 * @param sample Matrix pointer containing normalized 1x784 image data (0.0 to 1.0).
 * @param window_title Title bar text for the window.
 */
void render_mnist_sample(const Matrix *sample, const char *window_title);

#endif
