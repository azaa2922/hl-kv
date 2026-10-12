cmake_minimum_required(VERSION 3.20)

project(hlkv
  VERSION 0.1.0
  DESCRIPTION "Hetero-Latent ABI (HL-KV): heterogeneous KV-cache representation and execution"
  LANGUAGES CXX)

# ---- options -------------------------------------------------------------------------
option(HLKV_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(HLKV_BUILD_BENCHMARKS "Build Google Benchmark harnesses" ON)
set(HLKV_CUDA "AUTO" CACHE STRING
  "Build the CUDA kernels: AUTO (when a working nvcc >= 12 is found), ON (required) or OFF")
set_property(CACHE HLKV_CUDA PROPERTY STRINGS AUTO ON OFF)
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
# HLKV_CUDA_ENABLED (internal) is ON when the kernels are built.
set(HLKV_CUDA_ENABLED OFF)
string(TOUPPER "${HLKV_CUDA}" HLKV_CUDA)
if(NOT HLKV_CUDA MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "HLKV_CUDA must be AUTO, ON or OFF (got '${HLKV_CUDA}')")
endif()
if(NOT HLKV_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)  # finds nvcc and checks that it can build with the host compiler
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # sm_80 = A100, sm_89 = L4/L40S/RTX 40xx (first arch with FP8 cvt instructions),
      # sm_90 = H100/H200. Override with -DCMAKE_CUDA_ARCHITECTURES=...
      set(CMAKE_CUDA_ARCHITECTURES 80 89 90)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # One runtime for every target: libcudart.so (CUDA::cudart).
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
      set(HLKV_CUDA_ENABLED ON)
      message(STATUS "HL-KV: CUDA ${CUDAToolkit_VERSION} kernels enabled, architectures: "
                     "${CMAKE_CUDA_ARCHITECTURES}")
    else()
      set(_hlkv_cuda_reason "CUDA toolkit >= 12.0 not found (found: '${CUDAToolkit_VERSION}')")
    endif()
  else()
    set(_hlkv_cuda_reason "no working CUDA compiler (nvcc) found")
  endif()
  if(NOT HLKV_CUDA_ENABLED)
    if(HLKV_CUDA STREQUAL "ON")
      message(FATAL_ERROR "HLKV_CUDA=ON but ${_hlkv_cuda_reason}. Install CUDA 12+, set "
                          "CMAKE_CUDA_COMPILER, or use HLKV_CUDA=AUTO/OFF.")
    endif()
    message(STATUS "HL-KV: CUDA kernels disabled (${_hlkv_cuda_reason})")
  endif()
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
