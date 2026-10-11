# Test/benchmark dependencies.
#
# Resolution order:
#   1. An installed package (find_package CONFIG), e.g. from vcpkg, conda or a distro.
#   2. FetchContent from GitHub, pinned to an exact commit (not a movable tag).
#
# Offline builds: pass -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path> and
# -DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK=<path>, or install the packages.

include(FetchContent)

# googletest v1.15.2
set(HLKV_GOOGLETEST_COMMIT b514bdc898e2951020cbdca1304b75f5950d1f59)
# google/benchmark v1.9.1
set(HLKV_BENCHMARK_COMMIT c58e6d0710581e3a08d65c349664128a8d9a2461)

if(HLKV_BUILD_TESTS)
  find_package(GTest 1.14 CONFIG QUIET)
  if(GTest_FOUND)
    message(STATUS "HL-KV: using installed GoogleTest ${GTest_VERSION}")
  elseif(HLKV_FETCH_DEPENDENCIES)
    message(STATUS "HL-KV: fetching GoogleTest @ ${HLKV_GOOGLETEST_COMMIT}")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    # MSVC: link the CRT the same way as the rest of the project (/MD).
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG ${HLKV_GOOGLETEST_COMMIT})
    FetchContent_MakeAvailable(googletest)
  else()
    message(FATAL_ERROR "GoogleTest not found and HLKV_FETCH_DEPENDENCIES=OFF")
  endif()
endif()

if(HLKV_BUILD_BENCHMARKS)
  find_package(benchmark 1.8 CONFIG QUIET)
  if(benchmark_FOUND)
    message(STATUS "HL-KV: using installed Google Benchmark ${benchmark_VERSION}")
  elseif(HLKV_FETCH_DEPENDENCIES)
    message(STATUS "HL-KV: fetching Google Benchmark @ ${HLKV_BENCHMARK_COMMIT}")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG ${HLKV_BENCHMARK_COMMIT})
    FetchContent_MakeAvailable(googlebenchmark)
  else()
    message(FATAL_ERROR "Google Benchmark not found and HLKV_FETCH_DEPENDENCIES=OFF")
  endif()
endif()
