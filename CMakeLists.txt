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

# Use the C11 standard.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Export compile_commands.json for language servers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release builds if no build type is specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Collect all C source files.
#
# For larger projects, explicitly listing source files is recommended.
# GLOB_RECURSE is acceptable for small personal projects like this.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(SDL2 REQUIRED)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        SDL2::SDL2
)

# Place the executable in the project root.
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Public header directory.
target_include_directories(${PROJECT_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Compiler warnings.
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

# Enable AddressSanitizer and UndefinedBehaviorSanitizer in Debug builds.
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

# Enable Link Time Optimization (LTO/IPO) for Release builds if supported.
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

if(IPO_SUPPORTED)
    set_property(TARGET ${PROJECT_NAME}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()