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

# ---- CUDA kernels (HLKV_CUDA) ----------------------------------------------------------
# Compile flags every hl_quant kernel target uses:
#
#   --expt-relaxed-constexpr  constexpr helpers (format_traits) usable in device code
#   -lineinfo                 source correlation in Nsight Compute; no codegen change
#   -Xptxas=-v (optional)     per-kernel registers / spills / smem report
#
# Deliberately NOT used: --use_fast_math. It flushes subnormals to zero and
# approximates division, which would break bit-exact comparison against the
# CPU reference (FP8 subnormals and subnormal FP16 results are part of the contract).
if(HLKV_CUDA_ENABLED)
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

  # FP8 -> FP16 dequantization (first kernel; validated against the CPU reference).
  add_library(hl_quant_cuda src/dequant_kernel.cu)
  add_library(hl::quant_cuda ALIAS hl_quant_cuda)
  target_link_libraries(hl_quant_cuda
    PUBLIC hl::quant CUDA::cudart
    PRIVATE hl::quant_cuda_options)
  target_compile_features(hl_quant_cuda PUBLIC cxx_std_20)
  set_target_properties(hl_quant_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
