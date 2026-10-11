cmake_minimum_required(VERSION 3.20)

project(hlkv
  VERSION 0.1.0
  DESCRIPTION "Hetero-Latent ABI (HL-KV): heterogeneous KV-cache representation and execution"
  LANGUAGES CXX)

# ---- options -------------------------------------------------------------------------
option(HLKV_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(HLKV_BUILD_BENCHMARKS "Build Google Benchmark harnesses" ON)
option(HLKV_ENABLE_CUDA "Enable the CUDA toolchain for GPU kernels (requires CUDA 12+)" OFF)
option(HLKV_CUDA_VERBOSE_PTXAS "Pass -Xptxas=-v to report registers/spills per kernel" OFF)
option(HLKV_WARNINGS_AS_ERRORS "Treat compiler warnings in HL-KV targets as errors" OFF)
option(HLKV_FETCH_DEPENDENCIES "Fetch GoogleTest/Google Benchmark if not found locally" ON)
set(HLKV_SANITIZE "" CACHE STRING
  "GCC/Clang sanitizers for HL-KV targets, e.g. 'address,undefined' (empty = off)")

# ---- language standard -----------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++20, not gnu++20 (keeps FP contraction off by default)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_property(HLKV_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT HLKV_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "HL-KV: CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HlkvCompilerOptions)

# ---- optional CUDA toolchain -----------------------------------------------------------
if(HLKV_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "HLKV_ENABLE_CUDA=ON but no CUDA compiler (nvcc) was found. Install CUDA 12+ "
      "or set CMAKE_CUDA_COMPILER.")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # sm_80 = A100, sm_89 = L4/L40S/RTX 40xx (first arch with FP8 cvt instructions),
    # sm_90 = H100/H200. Override with -DCMAKE_CUDA_ARCHITECTURES=...
    set(CMAKE_CUDA_ARCHITECTURES 80 89 90)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  find_package(CUDAToolkit 12.0 REQUIRED)
  message(STATUS "HL-KV: CUDA ${CUDAToolkit_VERSION}, architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ---- projects ----------------------------------------------------------------------------
add_subdirectory(project1_allocator)
add_subdirectory(project2_quant)

if(HLKV_BUILD_TESTS OR HLKV_BUILD_BENCHMARKS)
  include(HlkvDependencies)
endif()
if(HLKV_BUILD_TESTS)
  enable_testing()
endif()
if(HLKV_BUILD_TESTS OR HLKV_BUILD_BENCHMARKS)
  add_subdirectory(tests)
endif()
