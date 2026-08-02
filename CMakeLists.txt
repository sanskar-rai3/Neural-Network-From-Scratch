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

# ==============================================================================
# CMake Requirements & Project Setup
# ==============================================================================

cmake_minimum_required(VERSION 3.16)

project(nn
    VERSION 0.1.0
    DESCRIPTION "Neural Network From Scratch in C"
    LANGUAGES C
)

# Enforce strict C11 standard without compiler-specific extensions (e.g., gnu11)
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
# Source Files & Dependencies
# ==============================================================================

# Automatically gather all C source files in src/ directory.
# CONFIGURE_DEPENDS re-runs CMake automatically when files are added/removed.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Define the target executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Locate required external dependency: SDL2
find_package(SDL2 REQUIRED)

# Link Math library (-lm) on non-MSVC toolchains (GCC/Clang)
if(NOT MSVC)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            m
    )
endif()

# Link SDL2 target
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        SDL2::SDL2
)

# ==============================================================================
# Target Configuration & Output
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
# Compiler Warnings & Diagnostics
# ==============================================================================

if(MSVC)
    target_compile_options(${PROJECT_NAME}
        PRIVATE
            /W4            # Warning Level 4
            /permissive-   # Strict standard conformance mode
    )
else()
    target_compile_options(${PROJECT_NAME}
        PRIVATE
            -Wall          # Enable common warnings
            -Wextra        # Enable extra helpful warnings
            -Wpedantic     # Enforce strict standard adherence
    )
endif()

# ==============================================================================
# Debug Build Options (Sanitizers)
# ==============================================================================

# Enable AddressSanitizer (ASan) & UndefinedBehaviorSanitizer (UBSan) in Debug
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
# Release Build Optimizations (Interprocedural / Link Time Optimization)
# ==============================================================================

# Enable Link Time Optimization (LTO) for Release builds if supported by toolchain
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

if(IPO_SUPPORTED)
    set_property(TARGET ${PROJECT_NAME}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()