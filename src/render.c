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
#include "render.h"
#include "matrix.h"

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdbool.h>

void render_mnist_sample(const Matrix *sample, const char *window_title) {
    if (!sample || sample->cols != 784) {
        fprintf(stderr, "Error: Invalid sample matrix for rendering (expected 1x784)\n");
        return;
    }

    /* Set nearest-neighbor scaling before creating textures to keep pixels sharp */
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");

    /* Initialize SDL Video Subsystem */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return;
    }

    /* Create Window */
    SDL_Window *window = SDL_CreateWindow(
        window_title,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_SIZE,
        WINDOW_SIZE,
        SDL_WINDOW_SHOWN
    );
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }

    /* Create Renderer */
    SDL_Renderer *renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!renderer) {
        fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    /* Create 28x28 Streaming Texture in 32-bit RGBA Format */
    SDL_Texture *texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        IMAGE_SIZE,
        IMAGE_SIZE
    );
    if (!texture) {
        fprintf(stderr, "SDL_CreateTexture error: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    /* -------------------------------------------------------------------------
     * Convert Normalized Floats (0.0 - 1.0) to RGBA8888 Pixel Buffer
     * ----------------------------------------------------------------------- */
    uint32_t pixels[IMAGE_SIZE * IMAGE_SIZE];

    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        /* Clamp normalized value to [0, 255] byte value */
        float val = sample->data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        
        uint8_t gray = (uint8_t)(val * 255.0f);

        /* Map gray value into RGBA channels: 0xRRGGBBAA */
        pixels[i] = (gray << 24) | (gray << 16) | (gray << 8) | 0xFF;
    }

    /* Update the streaming texture with raw pixel array */
    SDL_UpdateTexture(texture, NULL, pixels, IMAGE_SIZE * sizeof(uint32_t));

    /* -------------------------------------------------------------------------
     * Render Loop
     * ----------------------------------------------------------------------- */
    bool running = true;
    SDL_Event event;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        /* Clear Screen */
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        /* Copy texture to fill full scaled window dimensions */
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        /* Present rendered frame */
        SDL_RenderPresent(renderer);
    }

    /* Cleanup SDL Resources */
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
