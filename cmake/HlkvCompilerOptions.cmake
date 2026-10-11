# Warning / sanitizer configuration applied to HL-KV's own targets only
# (third-party dependencies keep their own flags).
#
# Flags are wrapped in $<COMPILE_LANGUAGE:CXX> so that targets which later gain
# .cu sources do not forward host-compiler flags to nvcc.

function(hlkv_configure_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /utf-8>
      # C4324: "structure was padded due to alignment specifier" — the alignment
      # is intentional (alignas(16)/alignas(32) ABI records).
      $<$<COMPILE_LANGUAGE:CXX>:/wd4324>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${HLKV_WARNINGS_AS_ERRORS}>>:/WX>)
    # Keep std::min/std::max usable if <windows.h> is ever pulled in.
    target_compile_definitions(${target} PRIVATE NOMINMAX)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
                                -Wshadow -Wold-style-cast -Wcast-align -Wnon-virtual-dtor
                                -Woverloaded-virtual -Wimplicit-fallthrough>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${HLKV_WARNINGS_AS_ERRORS}>>:-Werror>)
    if(HLKV_SANITIZE)
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${HLKV_SANITIZE} -fno-omit-frame-pointer
                                  -fno-sanitize-recover=all>)
      target_link_options(${target} PRIVATE -fsanitize=${HLKV_SANITIZE})
    endif()
  endif()
endfunction()
