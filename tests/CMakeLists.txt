if(HLKV_BUILD_TESTS)
  include(GoogleTest)

  # ---- Project 1: hl_allocator ----------------------------------------------------------
  add_executable(hl_allocator_tests
    project1/test_block_descriptor.cpp
    project1/test_hetero_block_table.cpp)
  target_link_libraries(hl_allocator_tests PRIVATE hl::allocator GTest::gtest_main)
  hlkv_configure_target(hl_allocator_tests)
  gtest_discover_tests(hl_allocator_tests DISCOVERY_MODE PRE_TEST)

  # ---- Project 2: hl_quant --------------------------------------------------------------
  add_executable(hl_quant_tests
    project2/test_dequant.cpp
    project2/test_quant_encode.cpp)
  target_link_libraries(hl_quant_tests PRIVATE hl::quant GTest::gtest_main)
  hlkv_configure_target(hl_quant_tests)
  gtest_discover_tests(hl_quant_tests DISCOVERY_MODE PRE_TEST)
endif()

if(HLKV_BUILD_BENCHMARKS)
  # The homogeneous baseline is compiled in its own translation unit so that
  # neither side of the comparison gets cross-TU inlining the other lacks.
  add_executable(hl_allocator_bench
    project1/bench_allocator.cpp
    project1/baseline_block_pool.cpp)
  target_link_libraries(hl_allocator_bench PRIVATE hl::allocator benchmark::benchmark)
  hlkv_configure_target(hl_allocator_bench)
endif()
