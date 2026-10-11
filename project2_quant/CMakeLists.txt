# Project 2 — hl_quant: KV storage formats and (de)quantization.
#
# Reference-first: the CPU reference (this library) is the numerical ground
# truth. GPU kernels are added only after the reference is tested, and are
# validated against it.

add_library(hl_quant
  src/quant_types.cpp
  src/cpu_dequant.cpp
  src/cpu_quant.cpp)
add_library(hl::quant ALIAS hl_quant)

target_include_directories(hl_quant
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(hl_quant PUBLIC cxx_std_20)
hlkv_configure_target(hl_quant)

# ---- optional CUDA toolchain contract ---------------------------------------------------
# No .cu sources exist yet. When HLKV_ENABLE_CUDA=ON this defines the compile
# flags every hl_quant kernel target must use, so the first kernel inherits a
# reviewed configuration:
#
#   --expt-relaxed-constexpr  constexpr helpers (format_traits) usable in device code
#   -lineinfo                 source correlation in Nsight Compute; no codegen change
#   -Xptxas=-v (optional)     per-kernel registers / spills / smem report
#
# Deliberately NOT used: --use_fast_math. It flushes subnormals to zero and
# approximates division, which would break bit-exact comparison against the
# CPU reference (FP8 subnormals are part of the format).
if(HLKV_ENABLE_CUDA)
  add_library(hl_quant_cuda_options INTERFACE)
  add_library(hl::quant_cuda_options ALIAS hl_quant_cuda_options)
  target_compile_options(hl_quant_cuda_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${HLKV_CUDA_VERBOSE_PTXAS}>>:-Xptxas=-v>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${HLKV_WARNINGS_AS_ERRORS}>>:--Werror=all-warnings>)
  if(MSVC)
    # Forward the host-compiler conformance flags through nvcc.
    target_compile_options(hl_quant_cuda_options INTERFACE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus,/permissive-,/utf-8>)
  endif()
  target_link_libraries(hl_quant_cuda_options INTERFACE CUDA::cudart)
  message(STATUS "HL-KV: hl_quant CUDA options target configured (no kernels yet)")
endif()
