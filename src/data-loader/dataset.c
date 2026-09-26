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

#include "data-loader/dataset.h"

#include <stdlib.h>

void dataset_create(Dataset *dataset, usize size, usize input_count, usize label_count) {
    dataset->size        = size;
    dataset->input_count = input_count;
    dataset->label_count = label_count;
    
    dataset->data   = malloc(size * input_count * sizeof(float));
    dataset->labels = malloc(size * label_count * sizeof(float));
    if (!dataset->data || !dataset->labels) {
        free(dataset->data);
        free(dataset->labels);
        
        dataset->data = NULL;
        dataset->labels = NULL;
    }
}

void dataset_destroy(Dataset* dataset) {
    free(dataset->data);
    free(dataset->labels);
    free(dataset);

    dataset->data = NULL;
    dataset->labels = NULL;
    dataset->size = 0;
    dataset->input_count = 0;
    dataset->label_count = 0;
}
