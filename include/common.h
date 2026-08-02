#ifndef __COMMON_H__
#define __COMMON_H__

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