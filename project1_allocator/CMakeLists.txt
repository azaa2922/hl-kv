# Project 1 — hl_allocator: host-side heterogeneous KV block table (no CUDA).

add_library(hl_allocator
  src/hetero_block_table.cpp
  src/status.cpp)
add_library(hl::allocator ALIAS hl_allocator)

target_include_directories(hl_allocator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(hl_allocator PUBLIC cxx_std_20)
hlkv_configure_target(hl_allocator)
