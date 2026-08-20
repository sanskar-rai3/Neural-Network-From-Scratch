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

#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stddef.h>

/*==============================================================================
 * Primitive Type Definitions
 *============================================================================*/

/**
 * @brief Unsigned 8-bit integer type (1 byte).
 */
typedef uint8_t u8;

/**
 * @brief Unsigned 16-bit integer type (2 bytes).
 */
typedef uint16_t u16;

/**
 * @brief Unsigned 32-bit integer type (4 bytes).
 */
typedef uint32_t u32;

/**
 * @brief Platform-native unsigned integer type used for matrix dimensions and indexing.
 */
typedef size_t usize;

#endif
