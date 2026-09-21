# Neural Network From Scratch in C
#
# Copyright (C) 2026 Sanskar Rai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.16)

project(nn
    VERSION 0.1.0
    DESCRIPTION "Neural Network From Scratch in C"
    LANGUAGES C
)

# ==============================================================================
# C Standard
# ==============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Generate compile_commands.json for clangd/LSP integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build configuration to Release if none is specified by the user
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ==============================================================================
# Source Files
# ==============================================================================

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

add_executable(${PROJECT_NAME} ${SOURCES})

# ==============================================================================
# Libraries
# ==============================================================================

if(NOT MSVC)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            m
    )
endif()

# ==============================================================================
# Target Configuration
# ==============================================================================

# Output binary directly to the project root directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Header include paths
target_include_directories(${PROJECT_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# ==============================================================================
# Compiler Warnings
# ==============================================================================

if(MSVC)
    target_compile_options(${PROJECT_NAME}
        PRIVATE
            /W4
            /permissive-
    )
else()
    target_compile_options(${PROJECT_NAME}
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

# ==============================================================================
# Assertions
# ==============================================================================

# Disable assert() in Release builds.
# assert() remains enabled in Debug builds.
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        $<$<CONFIG:Release>:NDEBUG>
)

# ==============================================================================
# Debug Build Options
# ==============================================================================

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
    target_compile_options(${PROJECT_NAME}
        PRIVATE
            -fsanitize=address
            -fsanitize=undefined
            -fno-omit-frame-pointer
    )

    target_link_options(${PROJECT_NAME}
        PRIVATE
            -fsanitize=address
            -fsanitize=undefined
    )
endif()

# ==============================================================================
# Release Build Optimizations
# ==============================================================================

include(CheckIPOSupported)

check_ipo_supported(
    RESULT IPO_SUPPORTED
    OUTPUT IPO_ERROR
)

if(IPO_SUPPORTED)
    set_property(
        TARGET ${PROJECT_NAME}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()
